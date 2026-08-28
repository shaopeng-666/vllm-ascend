#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash kimi-k3-c8-dp2tp16-serve.sh <local_hccl_ip> <dp_rank> [graph16|audit_graph32|eager]
#
# All infrastructure-specific values are supplied at runtime. Do not hard-code
# management IPs, HCCL IPs, usernames, model paths, or container paths here.

local_ip="${1:?local HCCL IP is required}"
dp_rank="${2:?DP rank is required}"
capture_mode="${3:-graph16}"

repo="${K3_REPO:?set K3_REPO inside the container}"
run_root="${K3_RUN_ROOT:?set K3_RUN_ROOT inside the container}"
model_path="${K3_MODEL_PATH:?set K3_MODEL_PATH inside the container}"
dp_address="${K3_DP_ADDRESS:?set K3_DP_ADDRESS to DP rank 0 service IP}"
socket_iface="${K3_SOCKET_IFACE:?set K3_SOCKET_IFACE}"
bind_host="${K3_BIND_HOST:?set K3_BIND_HOST to the service bind address}"
api_port="${K3_API_PORT:-8089}"
dp_rpc_port="${K3_DP_RPC_PORT:-13389}"
log_file="${run_root}/kimi3_dp2tp16_c8_${capture_mode}_rank${dp_rank}.log"

case "${capture_mode}" in
    graph16|mitigation)
        # With service max-num-seqs=32 and DP2, the expected per-DP capture
        # range ends at 16. Client concurrency above 16 must be confirmed from
        # dispatcher logs: excess requests should queue or execute eagerly.
        graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}')
        ;;
    audit_graph32|repro_graph32)
        # Audit-only mode. Size 32 is intentionally declared so startup and
        # dispatcher logs can show whether DP2 filtering removes it. Do not
        # infer that graph32 exists or was selected from this config alone.
        graph_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}')
        ;;
    eager)
        # A/B baseline: disable graph capture and replay.
        graph_args=(--enforce-eager)
        ;;
    *)
        echo "Unknown capture_mode=${capture_mode}; use graph16, audit_graph32, or eager" >&2
        exit 2
        ;;
esac

mkdir -p "${run_root}"
exec >"${log_file}" 2>&1

export PYTHONPATH="${repo}:${PYTHONPATH:-}"
export HCCL_IF_IP="${local_ip}"
export GLOO_SOCKET_IFNAME="${socket_iface}"
export TP_SOCKET_IFNAME="${socket_iface}"
export HCCL_SOCKET_IFNAME="${socket_iface}"
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# These variables enable the tracked host-readback probes. Their Python branches
# do not execute again during ACL Graph replay; absence of a probe message is
# therefore not accuracy evidence.
export VLLM_ASCEND_C8_NZ_DEBUG_FALLBACK_SITU=1
export VLLM_ASCEND_C8_NZ_DEBUG=1
export VLLM_ASCEND_C8_NZ_HOST_READBACK=1
export VLLM_ASCEND_C8_NZ_DEBUG_LAYER=model.layers.4.
export VLLM_ASCEND_C8_NZ_DEBUG_FIRST_STEPS=32
export VLLM_ASCEND_C8_NZ_DEBUG_EVERY=128
export VLLM_ASCEND_C8_NZ_DEBUG_PAIR=0,1
export VLLM_ASCEND_C8_NZ_DEBUG_PAIR_STEPS=8

dp_args=()
if [[ "${dp_rank}" != "0" ]]; then
    dp_args+=(--headless --data-parallel-start-rank "${dp_rank}")
fi

echo "capture_mode=${capture_mode} graph_args=${graph_args[*]} local_ip=${local_ip} dp_rank=${dp_rank}"
cd "${repo}"
exec vllm serve \
    "${model_path}" \
    "${dp_args[@]}" \
    --served-model-name kimi-k3 \
    --host "${bind_host}" \
    --port "${api_port}" \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-address "${dp_address}" \
    --data-parallel-rpc-port "${dp_rpc_port}" \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --max-model-len 8192 \
    --max-num-batched-tokens 24576 \
    --gpu-memory-utilization 0.98 \
    "${graph_args[@]}" \
    --mm-processor-cache-gb 0 \
    --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}' \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt '{"image":0}' \
    --enable-auto-tool-choice \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --language-model-only \
    --tokenizer-mode kimi_k3
