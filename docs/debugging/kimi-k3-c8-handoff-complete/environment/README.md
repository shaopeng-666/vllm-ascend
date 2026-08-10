# Environment snapshot status

旧运行环境的精确快照当前不可用。不要根据包内源码 commit 推断 CANN、driver、firmware、torch、torch-npu、vLLM 或镜像版本。

网络恢复后，在两台宿主机和两个容器分别运行 `scripts/collect_env_snapshot.sh`。原始快照可能包含主机名、IP、路径、镜像名和环境变量值；转交前必须先脱敏，同时保留版本号、镜像 digest、commit、quant config hash 和两机差异。

