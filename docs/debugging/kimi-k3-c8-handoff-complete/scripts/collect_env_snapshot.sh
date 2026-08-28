#!/usr/bin/env bash
set -euo pipefail

# Read-only collector apart from writing its report directory.
# Run once on each host/container before restarting the service.
# Raw output can contain infrastructure values; sanitize it before sharing.

repo="${K3_REPO:?set K3_REPO to the vllm-ascend checkout}"
out="${K3_ENV_OUT:?set K3_ENV_OUT to a new report directory}"
mkdir -p "${out}"

{
    date -Ins || date
    uname -a
    hostname
    printf 'whoami='; whoami
} >"${out}/system.txt" 2>&1

{
    python3 --version
    python3 -m pip --version
    python3 - <<'PY'
import importlib
for name in ("torch", "torch_npu", "vllm", "vllm_ascend"):
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', '<no __version__>')}")
    except Exception as exc:
        print(f"{name}=UNAVAILABLE ({type(exc).__name__}: {exc})")
PY
    python3 -m pip freeze
} >"${out}/python-packages.txt" 2>&1

{
    git -C "${repo}" rev-parse HEAD
    git -C "${repo}" status --short --untracked-files=all
    git -C "${repo}" submodule status --recursive || true
    git -C "${repo}" remote -v
} >"${out}/source-state.txt" 2>&1

git -C "${repo}" diff --binary >"${out}/tracked-working-tree.patch"
git -C "${repo}" diff --cached --binary >"${out}/tracked-index.patch"

{
    command -v npu-smi >/dev/null && npu-smi info || true
    command -v atc >/dev/null && atc --version || true
    find /usr/local/Ascend -maxdepth 3 -type f \
        \( -name version.info -o -name version.cfg \) -print -exec sed -n '1,120p' {} \; 2>/dev/null || true
} >"${out}/ascend.txt" 2>&1

{
    for key in \
        ASCEND_RT_VISIBLE_DEVICES HCCL_IF_IP GLOO_SOCKET_IFNAME \
        TP_SOCKET_IFNAME HCCL_SOCKET_IFNAME VLLM_ENGINE_READY_TIMEOUT_S \
        PYTORCH_NPU_ALLOC_CONF TASK_QUEUE_ENABLE HCCL_BUFFSIZE; do
        if [[ -v "${key}" ]]; then
            printf '%s=%s\n' "${key}" "${!key}"
        fi
    done
} >"${out}/selected-env.txt"

echo "Snapshot written to ${out}. Sanitize infrastructure values before sharing."

