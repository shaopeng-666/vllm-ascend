# Kimi K3 A3 C8 precision handoff package

生成时间：2026-08-10（Asia/Shanghai）

这是一份可离线移交给另一个 Codex 窗口的脱敏交付包。它包含已经恢复的精确 tracked 源码、Git bundle、补丁、复现客户端、GPQA 数据、现存历史结果、两机启动脚本、环境采集脚本和 SHA-256 校验。

重要边界：本包不是旧服务器目录的完整镜像。生成时旧、新两组服务器及服务端口从当前 Codex 均不可达，旧机的 dirty/untracked 文件、KDA 文件触发 probe、环境快照、DP2 case21 边界原始结果和 rank 日志无法补采。缺口逐项记录在 `MISSING-MATERIALS.md`。

## 先看什么

1. `STATUS-AND-NEXT-STEPS.md`：当前判断、实验边界和下一步优先级；
2. `REQUEST-COVERAGE.md`：对补充材料请求逐项映射已交付文件和缺口；
3. `source/RECONSTRUCTION.md`：精确 tracked 源码的两种恢复方式；
4. `scripts/kimi-k3-c8-dp2tp16-serve.sh`：`graph16`、`audit_graph32`、`eager` 三种服务模式；
5. `scripts/run_gpqa_stream_watch_liveprobe.py`：参数化 GPQA/SSE 客户端；
6. `results/RESULTS.md`：历史结果的证据边界；
7. `SHA256SUMS`：包内文件校验。

## 目录

```text
kimi-k3-c8-handoff-complete/
├── README.md
├── kimi-k3-c8-two-node-handoff.md
├── STATUS-AND-NEXT-STEPS.md
├── REQUEST-COVERAGE.md
├── MISSING-MATERIALS.md
├── SHA256SUMS
├── source/
│   ├── codex-c8-full-0065f5f3.tar.gz
│   ├── codex-c8-full-d37a76b-to-0065f5f3.patch
│   ├── codex-c8-full-f7dde85c8.bundle
│   ├── codex-c8-full-60462f479.bundle
│   ├── codex-c8-capture-guard.bundle
│   ├── codex-c8-host-readback-gate.bundle
│   ├── COMMIT-CHAIN.txt
│   ├── PROBES.md
│   ├── RECONSTRUCTION.md
│   └── UNTRACKED-STATUS.txt
├── scripts/
│   ├── kimi-k3-c8-dp2tp16-serve.sh
│   ├── kimi-k3-c8-dp2tp16-launch-host.sh
│   ├── run_gpqa_stream_watch_liveprobe.py
│   ├── test_kimi_kda_recurrent_ascendc_npu.py
│   ├── collect_env_snapshot.sh
│   ├── export_source_state.sh
│   ├── README.md
│   └── requirements.txt
├── data/
│   ├── GPQA_case21.jsonl
│   └── GPQA_diamond.partial_99.sha4e606bdc.jsonl
├── results/
│   ├── RESULTS.md
│   ├── dp2_c8_gpqa32_r1_t2048.sanitized.json
│   └── pairdiag_dp4/
└── environment/
    └── README.md
```

## 最短复现路径

在 Linux/NPU 环境中：

```bash
tar -xzf source/codex-c8-full-0065f5f3.tar.gz
chmod +x scripts/*.sh
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -r scripts/requirements.txt
export K3_API_URL='http://<rank0-service-address>:<api-port>/v1/chat/completions'
python3 scripts/run_gpqa_stream_watch_liveprobe.py \
  --data-path data/GPQA_case21.jsonl \
  --url "${K3_API_URL}" \
  --model kimi-k3 \
  --case-id 21 --concurrency 17 \
  --max-tokens 256 --dp-rank 0 \
  --continue-after-anomaly --unique-cache-salt \
  --output results/case21x17_dp0_salted_t256.json
```

注意：`GPQA_case21.jsonl` 保留原记录中的 `id: 21`，客户端按 record ID 查找，不是按文件行号查找。

## 两机服务

启动脚本不含真实基础设施值。每台宿主机设置：

```bash
export K3_CONTAINER='<container-name>'
export K3_HOST_RUN_ROOT='<host-run-directory>'
export K3_RUN_ROOT='<container-run-directory>'
export K3_REPO='<container-source-directory>'
export K3_MODEL_PATH='<checkpoint-directory>'
export K3_DP_ADDRESS='<rank0-dp-address>'
export K3_SOCKET_IFACE='<socket-interface>'
export K3_BIND_HOST='<service-bind-address>'
export K3_API_PORT='<api-port>'
export K3_DP_RPC_PORT='<dp-rpc-port>'
```

再分别以本机 HCCL 地址和 DP rank 调用宿主机脚本。当前按服务总并发 `--max-num-seqs 32`、DP2 计算，单 DP 域的 capture 上界为 16；客户端并发 17/32 时不能假定服务端同时 active 17/32，必须从 dispatcher 日志核对排队、capacity 和 graph/eager 选择。

## C8 on/off 的限制

恢复源码确认，C8/FAQuant 是 checkpoint 的 ModelSlim 量化描述驱动，不是一个已经确认存在的独立 serve 开关。严格四格 A/B 必须先比较并复制 checkpoint quant config，只禁用 attention 的 `fa_quant_type`/层级 FAQuant/C8 映射，同时保持共享专家 W4A8 等权重路径不变。本包没有模型权重和 quant config，因此不会提供一个可能误关其他量化路径的伪 C8-off 命令。

## 脱敏说明

- 文档和脚本不含原机器 IP、账号、容器名、业务绝对路径或认证信息；
- 历史结果中的原 API 地址已替换为保留域名 `api.example.invalid`；
- Git 源码归档和 bundle 是源码工件，可能包含公开仓库作者元数据、测试地址或公开 URL；它们不包含本次机器凭据；
- `data/` 是完整 GPQA 历史快照和 case21 子集，可能受数据集使用条款约束，请仅在授权范围内转交。

## 校验

Linux/macOS：

```bash
sha256sum -c SHA256SUMS
```

Windows PowerShell 可用 `Get-FileHash -Algorithm SHA256` 逐项比对。
