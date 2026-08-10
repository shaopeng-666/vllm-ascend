# Missing materials and recovery boundary

生成包时，当前 Codex 到旧两机、新提供两机以及服务端口均不可达。以下材料无法从本地缓存恢复，必须在网络恢复后从现场只读补采。

## 缺失材料

1. 旧 `codex-c8-full` 工作树的精确 `git status --short`、tracked dirty diff、全部 untracked 文件、当时 checkout 的 branch 名和 `git remote -v`；
2. 旧机未提交 KDA probe，尤其是包含下列符号的文件：
   - `VLLM_ASCEND_KDA_CACHE_PAIR_DEBUG`
   - `VLLM_ASCEND_CACHE_PAIR_TRIGGER_FILE`
   - `[KDAPAIR]`
3. 两台旧机的 vLLM rank0/rank1 服务日志和 launcher 日志；
4. DP2 case21 × 16/17/32 原始 JSON 结果；
5. 当时容器/镜像 digest、CANN、driver/firmware、Python、torch、torch-npu、vLLM、vllm-ascend 的完整环境快照；
6. 模型 checkpoint 的 quant config 脱敏副本，以及用户所说 C8-off 实验的准确切换方法；
7. 运行时 dispatcher 的真实 graph/eager 选择和 `batch_descriptor`；
8. graph replay 可用的持久 device-buffer probe；现有 tracked host-readback probe 不等价；
9. 新服务器上是否已有同一 checkout、模型和服务进程；当前不可达，不能判断。

## 本包提供的替代证据

- tracked 源码：精确恢复到 commit `0065f5f3bd88c1608932273dd9543d1919a55bbb`；
- Git 历史：四个 bundle 和完整 commit chain；
- 源码差异：`d37a76b...` 到 `0065f5f3...` 的 binary-safe patch；
- 数据：完整 99 条 GPQA snapshot 和 case21 原始记录；
- 历史结果：本地留存的 DP4 六轮 pairdiag 以及一个 DP2 GPQA32 结果，API 地址已脱敏；
- 客户端和启动脚本：已参数化，无真实基础设施值；
- 采集脚本：网络恢复后可补齐 source/environment 现场。

## 不应作出的声明

- 不要说旧工作树“完整恢复”，因为 untracked/dirty 状态缺失；
- 不要说 `KDAPAIR` probe 已包含或已启用；
- 不要把 capture list 当成运行时 graph32 分派的直接证据；
- 不要把 C8-off 简化为未经 checkpoint 配置核对的单一环境变量；
- 不要把 clean round 当作间歇性精度问题已经修复；
- 不要把下游多语言碎片或长 `!` 直接当作首错位置。
