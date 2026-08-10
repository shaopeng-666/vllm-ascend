from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import aiohttp


SCRIPT_RANGES = {
    "cjk": r"[\u3400-\u4dbf\u4e00-\u9fff]",
    "cyrillic": r"[\u0400-\u052f]",
    "arabic": r"[\u0600-\u06ff]",
    "devanagari": r"[\u0900-\u097f]",
    "hangul": r"[\uac00-\ud7af]",
    "thai": r"[\u0e00-\u0e7f]",
}


def load_cases(snapshot: Path, case_ids: list[int]) -> list[dict]:
    records = {}
    with snapshot.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records[record["id"]] = record
    missing = sorted(set(case_ids) - records.keys())
    if missing:
        raise RuntimeError(f"Cases absent from snapshot: {missing}")
    return [records[case_id] for case_id in case_ids]


def text_summary(text: str) -> dict:
    script_counts = {
        name: len(re.findall(pattern, text))
        for name, pattern in SCRIPT_RANGES.items()
    }
    bang_runs = [
        len(match.group(0)) for match in re.finditer(r"[!！]+", text)
    ]
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "replacement_count": text.count("\ufffd"),
        "longest_bang_run": max(bang_runs, default=0),
        "script_counts": script_counts,
        "exotic_script_total": sum(script_counts.values()),
    }


def strong_anomaly(text: str) -> tuple[bool, list[str]]:
    summary = text_summary(text)
    reasons = []
    if summary["replacement_count"] >= 2:
        reasons.append("replacement>=2")
    if summary["longest_bang_run"] >= 8:
        reasons.append("bang_run>=8")
    for name, count in summary["script_counts"].items():
        if count >= 10:
            reasons.append(f"{name}>=10")
    if summary["exotic_script_total"] >= 20:
        reasons.append("exotic_script_total>=20")
    return bool(reasons), reasons


async def run_one(
    session: aiohttp.ClientSession,
    case: dict,
    args: argparse.Namespace,
    stop_event: asyncio.Event,
) -> dict:
    role_map = {
        "HUMAN": "user",
        "USER": "user",
        "BOT": "assistant",
        "ASSISTANT": "assistant",
        "SYSTEM": "system",
    }
    messages = [
        {
            "role": role_map.get(item["role"].upper(), item["role"].lower()),
            "content": item["prompt"],
        }
        for item in case["origin_prompt"]
    ]
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.unique_cache_salt:
        payload["cache_salt"] = f"codex-liveprobe-{uuid.uuid4()}"
    headers = {}
    if args.dp_rank is not None:
        headers["X-data-parallel-rank"] = str(args.dp_rank)

    result = {
        "case_id": case["id"],
        "request_id": None,
        "reasoning": "",
        "content": "",
        "finish_reason": None,
        "error": None,
        "strong_anomaly": False,
        "anomaly_reasons": [],
        "first_anomaly_chars": None,
    }
    started = time.monotonic()
    try:
        async with session.post(args.url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                result["request_id"] = event.get("id", result["request_id"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    result["reasoning"] += (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or ""
                    )
                    result["content"] += delta.get("content") or ""
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]

                combined = result["reasoning"] + result["content"]
                is_bad, reasons = strong_anomaly(combined)
                if is_bad and not result["strong_anomaly"]:
                    result["strong_anomaly"] = True
                    result["anomaly_reasons"] = reasons
                    result["first_anomaly_chars"] = len(combined)
                    if args.trigger_on_anomaly:
                        Path(args.trigger_file).touch()
                        print(
                            f"PAIR_TRIGGER_CREATED path={args.trigger_file}",
                            flush=True,
                        )
                    if not args.continue_after_anomaly:
                        stop_event.set()
                    print(
                        "ANOMALY "
                        + json.dumps(
                            {
                                "case_id": case["id"],
                                "request_id": result["request_id"],
                                "first_anomaly_chars": len(combined),
                                "reasons": reasons,
                                **text_summary(combined),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if not args.continue_after_anomaly:
                        break
    except asyncio.CancelledError:
        result["error"] = "cancelled_after_peer_anomaly"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["elapsed_s"] = time.monotonic() - started
    combined = result["reasoning"] + result["content"]
    result["summary"] = text_summary(combined)
    print(
        "FINISH "
        + json.dumps(
            {
                "case_id": case["id"],
                "request_id": result["request_id"],
                "finish_reason": result["finish_reason"],
                "error": result["error"],
                "strong_anomaly": result["strong_anomaly"],
                "elapsed_s": round(result["elapsed_s"], 2),
                **result["summary"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--ids")
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--dp-rank", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trigger-on-anomaly", action="store_true")
    parser.add_argument("--continue-after-anomaly", action="store_true")
    parser.add_argument("--unique-cache-salt", action="store_true")
    parser.add_argument(
        "--trigger-file", default="/tmp/codex_cache_pair_trigger"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.ids is not None and args.case_id is not None:
        parser.error("use either --ids or --case-id/--concurrency, not both")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.case_id is not None:
        case_ids = [args.case_id] * args.concurrency
    elif args.ids is None:
        case_ids = list(range(32))
    elif "-" in args.ids and "," not in args.ids:
        first, last = (int(value) for value in args.ids.split("-", 1))
        case_ids = list(range(first, last + 1))
    else:
        case_ids = [int(value) for value in args.ids.split(",")]
    cases = load_cases(args.data_path, case_ids)

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)
    stop_event = asyncio.Event()
    started = time.monotonic()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        trust_env=False,
    ) as session:
        tasks = [
            asyncio.create_task(run_one(session, case, args, stop_event))
            for case in cases
        ]
        gather_task = asyncio.gather(*tasks)
        stop_task = asyncio.create_task(stop_event.wait())
        print(
            f"SUBMITTED count={len(tasks)} max_tokens={args.max_tokens} "
            f"dp_rank={args.dp_rank} url={args.url}",
            flush=True,
        )
        done, _ = await asyncio.wait(
            {gather_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        stopped_early = stop_task in done and stop_event.is_set()
        if stopped_early:
            for task in tasks:
                if not task.done():
                    task.cancel()
        results = await gather_task
        if not stop_task.done():
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)

    output = {
        "case_ids": case_ids,
        "data_path": str(args.data_path),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "dp_rank": args.dp_rank,
        "temperature": args.temperature,
        "url": args.url,
        "elapsed_s": time.monotonic() - started,
        "stopped_early": stopped_early,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    anomalies = [row["case_id"] for row in results if row["strong_anomaly"]]
    print(
        f"DONE stopped_early={stopped_early} anomalies={anomalies} "
        f"elapsed_s={output['elapsed_s']:.1f} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
