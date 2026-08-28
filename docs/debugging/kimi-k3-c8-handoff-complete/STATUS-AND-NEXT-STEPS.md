# Current status and next steps

## 已确认

- 历史两机拓扑是 DP2 / TP16 / EP，服务启动参数为 `--max-num-seqs 32`。本交接按服务总并发 32 处理，因此单 DP 域的 capture 上界是 `32 / 2 = 16`；仍需启动日志确认实际 scheduler 配置。
- graph 复现配置曾声明 `[1,2,4,8,16,32]`，但配置中的 32 不是 graph32 已生成/已选择的证据。在上述 DP2 口径下，32 应被过滤或不可达，客户端并发 17 时第 17 个请求应排队或回退 eager。
- 历史记录只能确认“固定请求到 DP0、客户端并发目标为 16/17/32”时的输出差异；没有证据证明服务端同时 active 17/32。
- 唯一 salt 使相同 prompt 的 prefix-cache block hash 跨请求隔离，所以不发生跨请求 prefix KV 命中；但 prefix cache 机制仍启用。
- 用户提供的 A/B：C8 off 正常；因此优先调查 C8 MLA 精度与缓存生命周期，而不是先假定 KDA 是根因。graph32 假设在单域 capture 上界 16 的口径下已经降级为待证伪的旧假设。
- 已恢复的最终 tracked commit 是 `0065f5f3bd88c1608932273dd9543d1919a55bbb`。
- tracked 探针包括 `C8REQ/C8NZCFG/C8NZDBG/C8NZMAP/C8NZPAIR`；不包括旧稿所述 `KDAPAIR`/trigger-file 符号。

## 尚未确认

- 没有运行时 dispatcher 日志证明 graph32 曾被生成或选择，也没有记录客户端并发 17 时服务端的 active/queued 分布。
- 没有严格完成 C8 on/off × graph16/eager 四格 A/B。
- 没有抓到第一个错误 layer/tensor/row/token。
- 现有 tracked probe 使用 Python host 分支和 readback，不能在 graph replay 每步执行；最新 commit 还在 capture 时关闭 host readback。
- `graph16`/`mitigation` 的 capture 上界是 16；客户端并发 17 时第 17 个请求理论上应排队或 eager，但仍需运行时分派日志和端到端精度验证。

## 下一步执行顺序

### P0：现场冻结

1. 不重启，先保存两机进程、服务日志、NPU 状态和容器启动时间；
2. 执行 `scripts/collect_env_snapshot.sh`；
3. 执行 `scripts/export_source_state.sh`，保留 HEAD、tracked diff、untracked 清单和 submodule；
4. 两机分别保存模型 quant config 的脱敏副本；
5. 确认两机源码和镜像 digest 一致。

### P1：调度边界与四格 A/B

先固定 case21、DP0、相同 max tokens/salt，分别发客户端并发 16/17/32，保存 active、queued、padded/capacity、最终 graph/eager 和 graph key。确认单域上界后，再完成 C8 on/off × graph16/eager 四格 A/B。

`audit_graph32` 只用于确认显式配置 size 32 后它是否被过滤，不能预先把它当作已知复现组合。所有格子保存：客户端 JSON、两机日志、dispatcher 选择、首异常字符位置、源代码和 quant config hash。

### P2：补运行时 graph 分派证据

在最终 dispatcher/runner 处仅记录一次：DP rank、active requests、unpadded/padded tokens、capacity、最终 `CudagraphMode`、`batch_descriptor`、graph key/cache hit。不能再仅凭 capture list 推断 16→17 的路径。

### P3：C8 MLA 首错探针

为目标 layer/step/row 预分配持久 device buffers；在图内无 host 分支地依次 copy：

1. `kv_c_normed`；
2. quantized latent KV（scatter 前）；
3. scatter 后真实 PA-NZ 物理 view；
4. FIA 的 `q_nope/q_pe`、block table、actual lengths、descale；
5. FIA output；
6. attention residual。

replay 完成后统一读回，比较 eager、graph16 和 BF16 reference。只有 dispatcher 确认 capacity 32 时，才把 row 16 和 rows 17–31 解释为 graph32 的 active/padding 行；否则 row 16 是 admission/回退边界。

### P4：按首错拆分

- scatter 前正常、cache 读回错：PA-NZ view/stride、slot mapping、block ownership、越界写；
- cache 正常、FIA 输出错：FIA C8 mode、descale、padded metadata、graph 生命周期；
- MLA 全正常、KDA 后首错：再恢复 KDA 主线，并以 CPU recurrence、eager AscendC、ACL Graph 三方比较。
