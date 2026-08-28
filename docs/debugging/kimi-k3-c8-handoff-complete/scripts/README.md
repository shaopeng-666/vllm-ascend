# Scripts

## `kimi-k3-c8-dp2tp16-serve.sh`

容器内启动 vLLM，支持 `graph16`（兼容别名 `mitigation`）、`audit_graph32`（兼容别名 `repro_graph32`）和 `eager`。基础设施信息（包括服务 bind address）全部从环境变量/位置参数读取。

当前按服务总并发 32、DP2 计算，单 DP 域的 capture 上界是 16。`audit_graph32` 故意把 32 放进配置，只用于核对它是否被过滤；不能把该模式名称当作 graph32 实际执行证据。客户端并发 17 时，第 17 个请求应排队或回退 eager，最终以 dispatcher 日志为准。

## `kimi-k3-c8-dp2tp16-launch-host.sh`

宿主机通过 `nohup docker exec` 启动容器内脚本。调用前必须显式设置容器、源码、模型、运行目录和网络变量。

## `run_gpqa_stream_watch_liveprobe.py`

客户端依赖标准库与 `aiohttp`。它重建 SSE 的 `reasoning/reasoning_content + content`，检测连续 `!`、replacement character 和不应大量出现的文字脚本，并可将请求固定到一个 DP rank。

`--unique-cache-salt` 给每个请求独立 UUID，从而隔离跨请求 prefix-cache block hash；不等价于关闭 prefix cache。

`--trigger-on-anomaly` 只会在客户端所在文件系统 touch 指定文件。恢复的服务源码没有读取该 trigger file 的 tracked 逻辑，所以除非后续接力者实现并确认共享挂载，该选项不会触发服务端 probe。

## `test_kimi_kda_recurrent_ascendc_npu.py`

这是从公开/恢复源码中留存的 KDA standalone 测试，不是旧机缺失的 `KDAPAIR` probe。历史尝试在 custom op 注册前失败，没有形成可用精度结论。重跑前必须：

1. 在正确 vllm-ascend/CANN 容器中完成 custom op 注册；
2. 先记录 DP2 下真实 admission/capacity；只有 dispatcher 证明 32 可达时才覆盖 `capacity=32, active=17`；
3. 同时比较 CPU recurrence、eager AscendC 和 ACL Graph；
4. 保存输入、输出、dtype/shape/stride、随机种子和环境 hash。

鉴于 C8-off 正常而 KDA 两侧都存在，这个测试当前是后置交叉验证，不是 P0 根因主线。

## `collect_env_snapshot.sh` / `export_source_state.sh`

网络恢复后用于现场补采。二者只写入调用者指定的新目录，但采集结果可能包含主机名、IP、路径、remote URL、环境变量和未跟踪秘密文件，转交前必须人工审核与脱敏。
