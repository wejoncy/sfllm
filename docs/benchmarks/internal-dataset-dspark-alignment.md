# SGLang 性能差距：原因与修正结果

主要配置问题：此前强制指定了 Triton draft attention，遗漏了可正常运行的官方 FlashInfer draft 路径。此路径修正不涉及 SGLang 源码改动。官方源码锁定 `e262b44eef1f2bf83e34345276e2bb91a086587e`，每次有效运行均通过前后 SHA/逐文件哈希与 clean worktree 检查。

## 已完成的300条正式结果

同一 target/draft、同一份内部数据集随机300条（seed42）、temperature0、输出上限1000、stop248044、BF16 activation、FP32 SSM；先预热，再清缓存。旧配置取原图对应的正式点。

| 并发 | 修正后配置 | 旧图 SG tok/s | 修正后 SG tok/s | 提升 | 接受长度 | 完成 |
|---:|---|---:|---:|---:|---:|---:|
| 12 | draft-flashinfer-w4 | 1776.65 | 2367.83 | 33.3% | 2.6853 | 300/300 |
| 64 | draft-flashinfer-w4 | 2134.11 | 3429.20 | 60.7% | 2.6925 | 300/300 |

## 控制草稿策略的诊断

以下短测是原样本前32条/C12与前128条/C64，不与上面的300条混为同一测量。完整命令与请求级结果保存在各配置目录。

| 配置 | C12 tok/s | C64 tok/s | C12 接受长度 | C64 接受长度 |
|---|---:|---:|---:|---:|
| 旧 Triton/width8/BF16 | 1734.82 | 2168.48 | 3.4362 | 3.3527 |
| draft-flashinfer | 2257.62 | 3278.10 | 3.4259 | 3.3426 |
| draft-flashinfer-fp8 | 2301.09 | 3251.95 | 3.4162 | 3.3216 |
| draft-flashinfer-w4 | 2305.22 | 3520.62 | 2.7360 | 2.7007 |
| draft-flashinfer-w6 | 2280.11 | 3413.28 | 3.2154 | 3.1178 |

同策略 SF（BF16 draft/full-vocab/width8）诊断：C12=3310.60 tok/s、accept3.4819；C64=4355.57 tok/s、accept3.3084。宽度4的正常接受长度约2.7；较短 verify 的有效接受长度不能直接与宽度8比较。

## 直接 profile 证据

独立 batch64 decode 图的草稿 attention kernel 时间总和：旧 Triton约40.75 ms/轮；官方 FlashInfer约4.26 ms/轮；SF约2.17 ms/轮。官方 FlashInfer的草稿图约7.87 ms/轮、target verify约26.24 ms/轮；SF完整循环图约27.17 ms/轮。

GDN 部分：SG约10.04 ms/轮、SF journal约7.49；conv：SG约2.48、SF约1.09。这里比较 kernel duration sums，node tracing 会增加开销，实时上下文长度也不是逐轮严格相等；不把这些数字替代无 profiler 的端到端吞吐或声称完整因果分解。捕获窗口 GPU kernel 活跃时间约95.4%（修正SG）与98.4%（SF），剩余差距不能主要归因于CPU空转；同步API时长也不等于CPU额外开销。

## 官方路径兼容性检查

- FA3：`flashattention_backend.py:231` 调用通用 resolver，而 `spec_info.py:298` 对 DSpark draft 无条件减1；`model_runner.py:875` 对 sample_from_anchor=false 保留完整宽度8。运行时接受长度约1.16–1.22，已拒绝。没有给运行中的官方 SGLang 打补丁。
- Compact adaptive：通过原生 JSON model override 声明 language_model_only=true 后解决 mRoPE reshape；完成B1/4/8/16/32/64各4种预算的实际成本拟合。官方GDN继承 `supports_ragged_verify_graph=False`，target退回eager；不含SIMULATE环境变量的新服务在真实128条内部数据集预热后段，`dspark_kv_inject -> project_target_hidden -> F.linear` 报CUDA/CUBLAS execution failed，整组排除。
- ReplaySSM：当前官方对GDN+DSPARK明确要求KDA，配置拒绝，不能假装已开启。

## 文件

- 各配置 `server-command.json`、`check-c*/bench/summary.json`、`formal-c*/bench/summary.json`：命令、统计与请求级结果。
- `cause-profile-sglang/steady.nsys-rep`：旧Triton路径。
- `cause-profile-sglang-flashinfer/steady.nsys-rep`：修正后官方路径。
- `cause-profile-sfllm/steady.nsys-rep`：SF同策略路径。
- profile目录 `kernel-analysis.json`：逐图、逐kernel时间。
- `corrected-results.json`：本次完整机器可读结果。

修正后的九点正式曲线现已完成，并替换 README 图片；完整结果见 [benchmark](internal-dataset-dspark.md)。SFLLM 沿用原九个已验证点，SGLang 每个并发点均按原生 FlashInfer draft 重新筛选。历史 Triton 曲线不再用作主图。
