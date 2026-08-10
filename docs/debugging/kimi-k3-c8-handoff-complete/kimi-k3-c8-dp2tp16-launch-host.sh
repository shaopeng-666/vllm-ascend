#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash kimi-k3-c8-dp2tp16-launch-host.sh <local_hccl_ip> <dp_rank> [graph16|audit_graph32|eager]

local_ip="${1:?local HCCL IP is required}"
dp_rank="${2:?DP rank is required}"
capture_mode="${3:-graph16}"

container_name="${K3_CONTAINER:?set K3_CONTAINER on the host}"
host_run_root="${K3_HOST_RUN_ROOT:?set K3_HOST_RUN_ROOT on the host}"
container_run_root="${K3_RUN_ROOT:?set K3_RUN_ROOT inside the container}"
serve_script="${container_run_root}/kimi-k3-c8-dp2tp16-serve.sh"
launcher_log="${host_run_root}/launcher_${capture_mode}_rank${dp_rank}.log"

: "${K3_REPO:?set K3_REPO inside the container}"
: "${K3_MODEL_PATH:?set K3_MODEL_PATH inside the container}"
: "${K3_DP_ADDRESS:?set K3_DP_ADDRESS to DP rank 0 service IP}"
: "${K3_SOCKET_IFACE:?set K3_SOCKET_IFACE}"
: "${K3_BIND_HOST:?set K3_BIND_HOST to the service bind address}"

mkdir -p "${host_run_root}"
nohup docker exec \
    -e K3_REPO \
    -e K3_RUN_ROOT \
    -e K3_MODEL_PATH \
    -e K3_DP_ADDRESS \
    -e K3_SOCKET_IFACE \
    -e K3_BIND_HOST \
    -e K3_API_PORT \
    -e K3_DP_RPC_PORT \
    "${container_name}" \
    bash "${serve_script}" "${local_ip}" "${dp_rank}" "${capture_mode}" \
    >"${launcher_log}" 2>&1 </dev/null &

echo "OUTER_PID=$! LAUNCHER_LOG=${launcher_log}"
