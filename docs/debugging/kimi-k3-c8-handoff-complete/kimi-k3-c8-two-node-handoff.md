# Kimi K3 A3 C8 两机精度问题接力交接（脱敏版）

更新时间：2026-08-10（Asia/Shanghai）

本文不包含真实管理 IP、HCCL IP、用户名、容器名、网卡名、模型目录或业务目录。所有基础设施信息都通过环境变量提供。更完整的源码、客户端、数据、历史结果和校验文件见 `kimi-k3-c8-handoff-complete/`。

## 1. 当前结论

- 两机配置曾成功运行 DP2 / TP16 / EP，每台使用 16 张 A3 NPU。
- 当前按“服务总并发 `--max-num-seqs 32`、DP2”解释：单个 DP 域的 capture 上界应为 `32 / 2 = 16`，capture size 范围是 `[1,16]`。这是本次交接采用的口径；仍需用启动日志中的最终 scheduler/compilation 配置确认实现是否确实按 DP 数切分。
- 原复现配置曾声明 capture sizes `[1,2,4,8,16,32]`，但配置里出现 32 不等于运行时真的生成或选择 graph32。在上述 DP2 口径下，size 32 应被过滤或不可达；第 17 个同时 active 的请求没有匹配图，应回退 eager，或者因为单域 admission 上限 16 而排队。
- 历史客户端观测是：case21 × 16 正常，客户端并发设为 17/32 时出现异常。旧结果没有保存 dispatcher 的实际 active/padded/capacity/mode，因此不能再把它表述成“active17 → graph32”的已证事实；典型输出仍是多语言碎片和长串 `!`。
- “每请求唯一 `cache_salt`”表示相同 prompt 不能跨请求共享同一个 prefix-cache block hash，因此不会跨请求命中相同 prefix KV；它不等同于关闭 prefix caching 的哈希、插入、淘汰和释放逻辑。
- 用户提供的强 A/B 是：不上 C8 时精度正常；因此当前主线是 **C8 MLA 精度/缓存生命周期**。graph32 假设因 DP2 单域 capture 上界 16 而显著减弱，必须先用 dispatcher 日志确认真实执行模式。KDA 可能放大上游错误，但目前没有证据把它列为首错点。
- 尚未获得第一处错误张量/算子证据，不能宣布根因。

## 2. 基础设施变量

在两台宿主机分别设置下列变量；不要把真实值提交进脚本或文档。

```bash
export K3_CONTAINER='<container-name>'
export K3_HOST_RUN_ROOT='<host-run-directory>'
export K3_RUN_ROOT='<container-run-directory>'
export K3_REPO='<container-source-directory>'
export K3_MODEL_PATH='<model-or-checkpoint-directory>'
export K3_DP_ADDRESS='<rank0-dp-service-address>'
export K3_SOCKET_IFACE='<socket-interface-name>'
export K3_BIND_HOST='<service-bind-address>'
export K3_API_PORT='<api-port>'
export K3_DP_RPC_PORT='<dp-rpc-port>'
```

另外分别记录两机的本地 HCCL 地址，作为启动脚本的第一个位置参数；DP rank 作为第二个位置参数。

## 3. 启动方式

容器内脚本：`kimi-k3-c8-dp2tp16-serve.sh`

宿主机脚本：`kimi-k3-c8-dp2tp16-launch-host.sh`

支持三种执行模式：

| 模式 | 行为 | 用途 |
|---|---|---|
| `graph16` / `mitigation` | 配置 `[1,2,4,8,16]` | DP2 正常配置；单域 capture 上界 16 |
| `audit_graph32` / `repro_graph32` | 故意声明 `[1,2,4,8,16,32]` | 审计 size 32 是否被过滤；不得预先称为 graph32 复现 |
| `eager` | `--enforce-eager` | C8 eager 对照 |

两机命令模板：

```bash
# DP rank 0
bash "${K3_HOST_RUN_ROOT}/kimi-k3-c8-dp2tp16-launch-host.sh" \
  '<rank0-local-hccl-address>' 0 graph16

# DP rank 1
bash "${K3_HOST_RUN_ROOT}/kimi-k3-c8-dp2tp16-launch-host.sh" \
  '<rank1-local-hccl-address>' 1 graph16
```

启动前先保存现场：两机的进程、容器启动时间、`git rev-parse HEAD`、`git status --short`、服务日志和 NPU 状态。服务若仍存活，不要先重启。

## 4. 客户端复现

完整包里的 `scripts/run_gpqa_stream_watch_liveprobe.py` 已参数化，需要显式提供数据和 API URL。

```bash
python3 scripts/run_gpqa_stream_watch_liveprobe.py \
  --data-path data/GPQA_case21.jsonl \
  --url "${K3_API_URL}" \
  --model kimi-k3 \
  --case-id 21 --concurrency 17 \
  --max-tokens 256 --dp-rank 0 \
  --continue-after-anomaly --unique-cache-salt \
  --output results/case21x17_dp0_salted_t256.json
```

建议依次运行并保存：

1. 单请求 smoke；
2. case21 × 16；
3. case21 × 17；
4. case21 × 32；
5. 32 个不同 GPQA case 的短复现。

边界实验应固定到同一个 DP rank，并同时记录 dispatcher 的真实 active 数、排队数、padded/capacity 和 graph/eager 选择。客户端并发 17/32 不是单 DP 同时 active 17/32 的证据；在 DP2、服务总并发 32 的口径下，单域最多应同时 admission 16。

## 5. 已知历史矩阵

| 实验 | 客户端并发目标 | Prefix 条件 | 最大输出 | 历史结果 |
|---|---:|---|---:|---|
| 单请求 smoke | 1 | 默认 | 64/2048 | 正常 |
| case21 请求固定 DP0 | 16 | 默认 | 512 | 0/16 异常 |
| case21 请求固定 DP0 | 17 | 每请求唯一 salt | 256 | 2/17 异常 |
| case21 请求固定 DP0 | 32 | 每请求唯一 salt | 512 | 16/32 异常 |
| case21 请求固定 DP0 | 32 | 默认 | 2048 | 32/32 异常 |

上述是历史客户端结果摘要，不是本次重新跑出的结果，也不证明服务端曾同时 active 17/32 或执行 graph32。旧机对应的 case21 × 16/17/32 原始 JSON、DP2 rank 日志和 dispatcher 证据当前不可达，完整包会明确列为缺口。

## 6. C8 开关的准确边界

恢复源码显示 ModelSlim 量化配置通过如下元数据选择路径：

- `fa_quant_type` 非空会设置 `enable_fa_quant`；
- `kv_cache_type == "C8"` 会设置 `enable_c8_quant`；
- MLA 的 INT8 K-cache dtype/FAQuant scheme 还取决于目标层是否被 `is_fa_quant_layer()` 选中。

所以“C8 off”不是一个已经确认存在的独立 serve 参数。若要在同一权重上做严格 A/B，必须先检查该 checkpoint 的量化描述，复制配置并只禁用 attention/FAQuant/C8 层映射，同时保持共享专家 W4A8 和其他权重配置不变。不能盲目全局删除一个字符串后就声称是等价 C8-off。

## 7. 已恢复探针及局限

精确 tracked 源码包含：

- `C8REQ`：请求数、request ID、scheduled/computed/prompt tokens、common-prefix blocks；
- `C8NZCFG`：latent/PE cache 的 shape、stride、dtype 和 format；
- `C8NZDBG`：scatter 后按 slot 读取的坏值计数/校验和；
- `C8NZMAP`：write slot 和 FIA slot 对照；
- `C8NZPAIR`：两行历史 cache 的 mismatch/校验和。

局限：这些探针包含 Python 分支和 host readback；最终版本还明确在 graph capture 时禁用 host readback。它们不能证明 replay 每步的数据正确，也不是“持久 device debug buffer + replay 后统一读回”的完整实现。

交接旧稿曾提到 `VLLM_ASCEND_KDA_CACHE_PAIR_DEBUG`、`VLLM_ASCEND_CACHE_PAIR_TRIGGER_FILE` 和 `[KDAPAIR]`，但这些符号不在已恢复的 tracked 树中。它们很可能属于旧机未提交改动；当前不可把它们当成可用探针。

## 8. 下一步排查计划

### P0：恢复现场并冻结版本

- 运行完整包的 `scripts/collect_env_snapshot.sh`；
- 保存两机源代码 commit、dirty diff、untracked 文件清单、submodule、镜像 digest、CANN/torch/torch-npu/vLLM/vllm-ascend 版本；
- 保存模型量化配置的脱敏副本；
- 运行时直接记录 dispatcher 的 active requests、padded/capacity、最终 graph/eager 选择和 batch descriptor。

### P1：先证实调度边界，再做 C8 四格 A/B

固定 case21、DP0、相同输出长度与 salt 条件，先运行客户端并发 16/17/32，并记录服务端真实 active、queued、padded/capacity 和最终 graph/eager。预期 DP2 单域 capture 上界为 16；客户端并发 17 时，第 17 个请求应排队或走 eager，不能预设 graph32。

在真实模式确认后，对 C8 on/off 分别运行 `graph16` 和 `eager`。`audit_graph32` 只用来验证 size 32 是否被过滤，不再作为四格中的既定可执行路径。C8-off 必须记录它和 C8-on 的 checkpoint/config 差异。

### P2：在第一个 C8 MLA 层抓首错

同一 layer/step/row 依次比较：

1. `kv_c_normed`（量化前）；
2. quantized latent KV（scatter 前）；
3. scatter 后按真实 PA-NZ view 读回并反量化；
4. FIA 的 `q_nope/q_pe`、block table、actual lengths 和全部 descale；
5. FIA 输出；
6. attention residual 输出。

必须使用预分配、持久的 device debug buffer 和无 host 条件分支的 device copy，让 copy 节点进入图；在 replay 完成后统一读回。先按 dispatcher 证据选择有效行：若单域只 admission 16，则 row 16 是排队/回退边界，不应假定它属于 graph32；只有日志证明 capacity 32 时才检查 rows 16–31 的 graph padding。

### P3：根据首错拆算子

- scatter 前正常、读回错误：查 PA-NZ view/stride、slot mapping、block ownership 和越界写；
- cache 读回正常、FIA 输出错误：查 C8 FIA、descale、padded metadata 和 graph replay 生命周期；
- 所有 MLA 边界正常、KDA 后首次错误：再把 KDA/recurrent state 恢复为主线。

## 9. 当前可达性与证据边界

本交接生成时，从当前 Codex 环境无法连接旧两机、新提供两机或服务端口。因此：

- 无法判断远端容器和 vLLM 当前是否仍运行；
- 无法补采旧机 untracked KDA probe、dirty diff、环境快照和缺失日志；
- 无法在 NPU 环境执行新的四格 A/B；
- 无法从本机直接复制到另一台电脑的目标目录。

完整包保留了能离线恢复的精确 tracked Git commit/bundle、参数化脚本、数据和已有结果，并通过 `MISSING-MATERIALS.md` 明确列出剩余材料。
