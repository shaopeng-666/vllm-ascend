# 补充交接材料覆盖表

本表对应另一 Codex 任务提出的补充材料要求。原请求中出现的机器 IP、账号、容器名、模型绝对路径和目标电脑目录不在本包中复述；所有运行环境值均以占位符或环境变量表示。

| 请求项 | 本包材料 | 状态与边界 |
|---|---|---|
| `codex-c8-full` tracked 源码现场 | `source/codex-c8-full-0065f5f3.tar.gz`、四个 Git bundle、binary-safe patch、`COMMIT-CHAIN.txt`、`RECONSTRUCTION.md` | 已交付 tracked tree 与可恢复历史；旧机 dirty/untracked、原 branch/remote 精确值缺失 |
| 基准 commit、HEAD、提交链、submodule | `source/COMMIT-CHAIN.txt`、`source/RECONSTRUCTION.md` | 已记录；submodule 内容未打入归档 |
| C8/KDA/PR #13225 修改分类 | `source/PROBES.md`、`source/UNTRACKED-STATUS.txt` | C8 tracked probe 已定位；旧稿所述 KDA trigger probe 未恢复 |
| 参数化复现客户端与依赖 | `scripts/run_gpqa_stream_watch_liveprobe.py`、`scripts/requirements.txt`、`scripts/README.md` | 已交付；API、数据和输出路径不含真实机器值 |
| GPQA case21 与 SHA-256 | `data/GPQA_case21.jsonl`、`SHA256SUMS` | 已交付；另附历史 99 条 snapshot |
| smoke、16/17/32 路命令 | `kimi-k3-c8-two-node-handoff.md`、`README.md`、客户端 `--help` 参数 | 已交付模板；16/17/32 是客户端并发目标，不等同服务端同时 active 数 |
| `X-data-parallel-rank`、`cache_salt`、异常检测 | `scripts/README.md` 与客户端源码 | 已说明；unique salt 只隔离跨请求 prefix hash，不关闭 cache 管理 |
| C8 on/off 精确切换 | `kimi-k3-c8-two-node-handoff.md` 第 6 节、`source/PROBES.md` | 已定位到 checkpoint ModelSlim quant metadata；因缺 quant config，不能伪造单一 C8-off 开关 |
| eager/graph capture 切换 | 两机启动脚本、`scripts/README.md` | `eager` 使用 `--enforce-eager`；`graph16` 是 DP2 主配置；`audit_graph32` 只审计 size 32 是否被过滤 |
| active17/graph32 运行时证据 | `STATUS-AND-NEXT-STEPS.md` P1/P2 | 当前缺失；按服务总并发 32、DP2，单域 capture 上界应为 16，客户端第 17 请求应排队或 eager |
| C8 probe 入口、触发与限制 | `source/PROBES.md` | 已交付 tracked probe 定位；现有 host-readback 不能代表 replay 每步采样 |
| 持久 device-buffer 首错方案 | `kimi-k3-c8-two-node-handoff.md` P2、`STATUS-AND-NEXT-STEPS.md` P3 | 提供设计要求；尚未实现和验证 |
| 历史最小证据集 | `results/RESULTS.md`、`results/pairdiag_dp4/`、DP2 sanitized JSON | 已交付本地留存结果；旧 DP2 case21 边界原始 JSON/rank 日志缺失 |
| 旧环境版本快照 | `scripts/collect_env_snapshot.sh`、`environment/README.md` | 采集脚本已交付；旧两机真实快照因不可达而缺失 |
| 源码现场导出 | `scripts/export_source_state.sh` | 已交付；网络恢复后应先只读运行 |
| 启动脚本转义损坏修复 | `scripts/kimi-k3-c8-dp2tp16-serve.sh` | `log_file="${run_root}/..."` 已恢复，并通过 `bash -n` |
| 全文件校验 | `SHA256SUMS` | 每次文档或脚本更新后重新生成并验证 |
| 安全要求 | 全包扫描、参数化脚本、sanitized 结果 | 不含已知机器 IP、账号、容器名、私钥、token、cookie 或 API key；公开源码中的示例 token/URL 不代表现场凭据 |

完整不可恢复项见 `MISSING-MATERIALS.md`。这些缺口不得在后续交接中被描述为“已经补齐”或“已验证”。
