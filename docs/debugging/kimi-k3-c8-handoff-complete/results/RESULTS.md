# Historical results

这些文件是本地留存历史证据，不是本次重新运行的结果。原私网 API 地址已替换为 `api.example.invalid`，其余响应内容和元数据未主动改写。

## DP4 六轮 GPQA32

`pairdiag_dp4/` 包含六轮 JSON 和循环日志。每轮 32 个不同 GPQA case、`max_tokens=2048`，未固定 DP rank，因此在 DP4 服务上通常会分摊请求，不能直接当作单 DP active32 边界实验。

| 轮次 | 结果 |
|---:|---|
| 1 | 无 strong anomaly |
| 2 | 无 strong anomaly |
| 3 | case25 首次异常，约 char 3497 |
| 4 | case20 首次异常，约 char 83 |
| 5 | case25 首次异常，约 char 36 |
| 6 | case24 首次异常，char 2686 |

这证明问题具有状态/历史相关和间歇性；两轮 clean 不代表修复。

## DP2 GPQA32

`dp2_c8_gpqa32_r1_t2048.sanitized.json` 是两机 DP2/C8 服务的一轮 32 个不同 case 结果。首个 strong anomaly 是 case21，记录的 `first_anomaly_chars=4469`，触发原因为 `cjk>=10`。该文件随后取消 peers，因此不能用“未完成的其他请求”计算整体异常率。它保留了第一异常附近的完整 reasoning/content，适合做输出层症状比对，不足以定位首错算子。

## 缺失的关键边界结果

旧机曾记录：case21 请求固定 DP0 时，客户端并发目标 16 默认为 0/16；17 唯一 salt 为 2/17；32 唯一 salt 为 16/32；32 默认、2048 tokens 为 32/32。但这些 DP2 原始 JSON、rank 日志和 dispatcher 状态当前不在本地包中，不能把客户端并发目标等同于服务端同时 active 数，也不能把表格摘要当成 graph32 证据。网络恢复后必须补采或重跑。
