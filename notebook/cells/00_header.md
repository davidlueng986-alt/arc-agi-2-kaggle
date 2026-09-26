# ARC Prize 2026 (ARC-AGI-2) — H1 "Pure Reasoning Ceiling"

**Branch** `h1/pure-xhigh` · **Kernel** `liangwanyiudavid/arc26-h1-pure-xhigh`

Hypothesis under test: **Qwen3.8-27B FP8 + `reasoning_effort=xhigh` + a minimal harness
reaches ≥ 50% on ARC-AGI-2**, with no judge model, no code-refinement loop, no test-time
training and no fine-tuning. The only levers are reasoning effort, sample count, CSV grid
encoding, tolerant parsing and shape-diverse two-attempt selection.

Full quantitative budget model, score prediction and falsification criteria: `HYPOTHESIS.md`.

### What this kernel does

1. Writes the `harness/` package to disk (stdlib-only, @TESTS@ test functions green offline).
2. Installs vLLM — from a pre-baked wheelhouse if attached, else `pip` (needs internet, i.e. a non-rerun save).
3. Launches one vLLM instance, `--tensor-parallel-size 4`, through a **6-rung flag fallback ladder** so a version-sensitive flag cannot kill the run.
4. Probes `reasoning_effort=xhigh` and **calibrates** the per-sample token cost from real output.
5. **Phase A:** 2 × `low`-effort samples on *all* tasks (~3.7% of budget, ~23 min) → a complete, schema-valid `submission.json` exists from that point on.
6. **Phase B:** xhigh **waves** over all tasks (never task-by-task — serial execution would overrun 12 h by ~1.9×), with consensus early-stop and a governor that re-plans every wave against *measured* throughput.
7. Cumulative atomic submission write after every wave + an independent watchdog thread.

### Attachments required

| Type | Slug | Why |
|---|---|---|
| Competition | `arc-prize-2026-arc-agi-2` | provides `arc-agi_test_challenges.json` at rerun |
| Dataset | `saltb0x/qwen3-8-27b-fp8` | 25.3 GB official FP8 HF checkpoint (verified layout) |
| Dataset *(optional)* | `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` | interchangeable FP8 mirror; auto-detected as a fallback |
| Dataset *(optional)* | a pre-baked vLLM wheelhouse | **required for the offline competition rerun** |

> **Do NOT use `sergiodefreitas/qwen3-8-27b`** — it is an Ollama GGUF split archive
> (`ollama_weights.part_aa`, …), not an HF checkpoint. vLLM cannot load it.
> **Do NOT use an NVFP4 mirror** — NVFP4 is Blackwell-native; L4 is Ada (sm_89).

### Metadata

`enable_gpu` is **`false`** in `kernel-metadata.json`: this kernel was authored and pushed
without executing it, because running it consumes paid L4x4 quota. **Enable the L4x4
accelerator in the Kaggle UI before running.** In non-rerun mode it scores itself against
the 120-task evaluation split; in competition rerun it uses the 240 unseen test tasks.
