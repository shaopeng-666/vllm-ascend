# Recovered tracked probes

以下位置以 commit `0065f5f3bd88c1608932273dd9543d1919a55bbb` 为准。行号只对应该 commit。

## 修改分类

| 类别 | 恢复证据 |
|---|---|
| PR #13225/Kimi K3 主线 | `2c2b785` 到 `9bb9f4e` 的提交链；包含 A3 C8 MLA、PA-NZ/FIA、共享专家 W4A8 及相关修复 |
| C8 diagnostics | `33b7265`；主要修改 `mla_v1.py` 和 `model_runner_v1.py`，增加下述 C8NZ/C8REQ 日志 |
| 测试/诊断修正 | `8ea6fb0`、`60462f4`、`f7dde85` |
| graph capture/readback gate | `0a6742d`、`0065f5f3` |
| KDA probe | 恢复的 tracked 树中没有 `KDAPAIR`/trigger-file probe；不能归到任何 tracked commit |

## 请求元数据

文件：`vllm_ascend/worker/model_runner_v1.py`

- `execute_model()`：约 line 1710；
- `[C8REQ]`：line 1861；
- 输出 DP rank、debug step、request 数/ID、scheduled/computed/prompt tokens、common-prefix blocks 和 prefix-cache 指标。

局限：它没有记录最终 `CudagraphMode`、capacity、排队状态或 `batch_descriptor`，所以不能证明客户端并发 17 对应服务端同时 active 17，更不能证明选择了 graph32。按服务总并发 32、DP2 的当前口径，单域 capture 上界应为 16。

## C8 scatter 写后核对

文件：`vllm_ascend/attention/mla_v1.py`

- `_exec_kv_no_rope()`：line 1686；
- `kv_c_normed`：line 1705；
- `torch_npu.npu_quantize()`：line 1720；
- latent PA-NZ 5D view：line 1733，最后一维 32；
- positional PA-NZ 5D view：line 1740，最后一维 16；
- `[C8NZCFG]`：line 1764；
- 两次 `npu_scatter_pa_kv_cache(..., cache_mode="PA_NZ")`：line 1787/1795；
- `[C8NZDBG]`：line 1836，对有效 slot 比较 scatter 后 cache 与输入。

门控条件：`VLLM_ASCEND_C8_NZ_DEBUG=1`、`VLLM_ASCEND_C8_NZ_HOST_READBACK=1`、目标 layer 命中、TP group rank 0，并且 line 1803 要求当前 stream 不在 capture。

## FIA slot mapping 与两行 history

文件：`vllm_ascend/attention/mla_v1.py`

- `_forward_decode()`：line 1977；
- `[C8NZMAP]`：line 2020，根据 `seq_lens` 和 `block_table` 重算 FIA slot，与记录的 write slot 比较；
- A3 FAQuant latent/PE cache view：line 2041–2048；
- `_history()`：line 2096，按 physical block + offset 收集一行完整历史；
- `[C8NZPAIR]`：line 2137，比较环境变量指定的两行并输出 mismatch/校验和。

同样要求 `not torch.npu.is_current_stream_capturing()`，并大量调用 `.cpu()`/`.item()`。因此这些逻辑是 eager/capture 外 host-readback 诊断，不是每次 graph replay 内的 probe。

## 共享专家 situ fallback

文件：`vllm_ascend/ops/fused_moe/moe_mlp.py`

- `_w4a8_situ_apply_mlp()`：line 120；
- `VLLM_ASCEND_C8_NZ_DEBUG_FALLBACK_SITU`：line 210。

这是诊断 fallback，不等价于对 C8 MLA 首错的证明，也不应和未恢复的 KDA trigger probe 混淆。

## C8/FAQuant 配置入口

文件：`vllm_ascend/quantization/modelslim_config.py`

- `is_fa_quant_layer()`：line 799；
- `get_kv_quant_dtype()`：line 831，MLA FAQuant 在 A3 使用 INT8 K cache；
- `_add_kvcache_quant_metadata()`：line 1023；
- `fa_quant_type` 非空设置 `enable_fa_quant`；
- `kv_cache_type == "C8"` 设置 `enable_c8_quant`。

严格 C8-off A/B 要检查 checkpoint 的完整量化描述及 layer keys，不能只依据服务命令猜开关。

## 缺失 probe

在恢复的所有 tracked commit 中均未发现：

```text
VLLM_ASCEND_KDA_CACHE_PAIR_DEBUG
VLLM_ASCEND_CACHE_PAIR_TRIGGER_FILE
[KDAPAIR]
```

旧交接中对这些符号的描述只能视为“旧工作树可能存在的 untracked/dirty 实验代码”，不能视为本包能力。
