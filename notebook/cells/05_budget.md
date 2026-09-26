# Compute budget — the numbers this kernel schedules against

Reproduced live by `python scripts/eval_local.py --budget-table`.

**Hardware** 4 × L4: 24 GB, ~300 GB/s, sm_89 Ada, **no NVLink** → TP all-reduce is latency-bound.
**Model** Qwen3.8-27B FP8 ≈ 27 GB → TP=4 → **6.75 GB weights/GPU**.
**KV** only 16 of 64 layers are full attention (4 KV heads, head_dim 256):
`16 × 4 × 256 × 2 × 2 B = 64 KB/token` aggregate = **16 KB/token/rank**.

### Memory (per GPU, TP=4)

```
24.0 × 0.92 util          = 22.08 GB
− 6.75 weight shard       = 15.33 GB
− ~2.0 ctx/activ/DeltaNet = 13.33 GB KV pool
13.33e9 / 16,384 B        = 813,600 tokens/rank
worst seq = 4,255 prompt + 16,384 completion = 20,640 → 39 concurrent
mean  seq = 1,135 prompt + 10,000 completion = 11,135 → 73 concurrent
```
→ `--max-num-seqs 64`, harness `concurrency=48`. Memory is **not** the constraint.

### Decode step (per rank): `22.5 ms weights + B·L̄·16 KB/300 GB/s + ~6 ms collectives`

| B | L̄ | step | raw tok/s | derated ×0.55 | BW/rank |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 K | 31.9 ms | 251 | 138 | 51% |
| 32 | 8 K | 42.1 ms | 760 | 418 | 87% |
| 48 | 11 K | 54.1 ms | 887 | 488 | 95% |

→ **planning figures: 280 (pessimistic) / 450 (central) / 700 (optimistic) tok/s aggregate.**
Cross-check: 450 tok/s = 24.3 TFLOPS = **2.5% MFU** of 968 TFLOPS FP8 peak — decode is
memory-bound, so a low MFU is expected and confirms we are not compute-limited.

### Wall clock

```
43,200 s (12 h) − 600 submission buffer − 120 finalize reserve = 42,480 s  (t_gen)
             − 1,020 s startup/load/compile                     = 41,460 s  usable
             − 10% safety reserve (4,146 s)                     = 37,314 s  planned window
```

### Tokens and the answer to the crux question

| Scenario | tok/s | completion budget | tok/task | **xhigh samples/task** | total run |
|---|---:|---:|---:|---:|---:|
| Pessimistic | 280 | 10.45 M | 43,533 | **4.09** | 10.36 h |
| Central | 450 | 16.79 M | 69,964 | **6.74** | 10.35 h |
| Optimistic | 700 | 26.12 M | 108,832 | **10.62** | 10.37 h |

Prompt cost is measured, not guessed: **test split mean 1,135 tokens** (p90 2,067, max 4,255);
evaluation split mean 1,946 (max 5,379). With prefix caching, total prefill ≈ 76 s = 0.2% of
the window. Worst case 4,255 + 16,384 = 20,639 < 32,768 → `--max-model-len 32768` is safe.

**Phase A** = `240 × 2 × 1,300 = 624,000` tokens = **3.7%** of budget = **23 min**.
**Phase B** = central `(16.79 M − 0.62 M) / 240 / 10,000` = **6.74 xhigh samples/task**.

### Why waves and not task-by-task

Serial scheduling keeps only ~6.7 requests in flight → step 30.5 ms → **230 tok/s** →
16.17 M tokens would take **19.5 h**. ✗ **Serial execution of this exact hypothesis overruns
the budget by ~1.9×.** Waves over all tasks at `concurrency=48` are what make B≈40 and
therefore 450 tok/s reachable. This is a precondition of the hypothesis, not an optimisation.
