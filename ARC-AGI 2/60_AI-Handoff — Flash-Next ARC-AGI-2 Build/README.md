# ARC-AGI-2 executable-program induction — checkpoint 06

Implemented: ARC loader → model-generated Python → sandbox per grid → exact training verifier → structured failure → repair/multiple roots → lineage/dedup → diverse two-output selection → reproducible evaluation/submission.

**Flash now runs and generates executable Python on L4×4. A high-scoring ARC solver has not yet been demonstrated.** The measured eager baseline produced no ARC candidates before its request deadlines. Full120 public evaluation and neural search/repair ablations remain unmeasured.

## Authoritative measured evidence

G04, immutable Kaggle v13, source `d952c0b1e667a07fb0a95395858ab179c430552bcfc70f6e1cb1e5b952ce3bd2`:

| Measurement | Result |
|---|---|
| Model | Flash-Next W4A16 + FP8PLE, Kaggle model version1 |
| Hardware | 4×L4, each23034MiB |
| Startup | 913.107s |
| Sampled VRAM peak, 5s polling | 20959MiB/rank =20.468GiB/rank |
| Real generated-code smoke | Passed;61 completion tokens,18.867s; sandbox row reversal exact |
| A1 benchmark | Fixed4 public tasks,6 test outputs; one program/no repair |
| pass@1 / pass@2 | 0/6 /0/6 |
| Exact-train-fit | 0/4 |
| Candidates / repairs | 0 /0 |
| Failures | 4 generation TimeoutError, each480s |
| Conditional generalization | Undefined; no train-fitting candidate |
| Logged steady generation throughput | Median6.3tok/s,191 rolling samples≥5tok/s |
| End-to-end GPU device-hours | 3.17095; verified correct outputs/hour0 |
| Main / complete notebook log duration | 2853.860s /2880.196s |

At6.3tok/s,8192 decode tokens would require about1300s; this arithmetic explains why a480s request cap cannot reliably produce final code. It does not establish the model's unconstrained ARC capability or the quality cost of quantization. A4096 thinking budget alone would still need about650s on this eager path, so it was not promoted as a sufficient fix.

G01/G02/G03 failure logs and exact sources remain preserved. 27B weights fit but readiness was not achieved within earlier limits; no27B model score is claimed. G04 actual smoke, server statistics, configs, calls and metrics are under experiments/G04-output. See RESEARCH_LOG.md and EXPERIMENT_REGISTER.json.

## Current experimental candidate and Kaggle status

CPU v20, source `7850d97d39ceeb57fb8989467c122075f425ddbe9074de9af6d96104c6094592`, passed24 Kaggle tests (23 pass,1 host-only deployment test skipped),9.534s. All24 host tests passed5.579s.

Candidate: same Flash checkpoint/precision/CPU PLE/32K/TP4+EP/.92 memory/prefill512, allow default vLLM compilation/CUDA graphs,2 serving slots,2 concurrent task workers, streaming,4096 thinking cap, A1 policy. It is a combined feasibility experiment, not an isolated causal ablation or a proven quality/speed improvement. GPU performance is unmeasured.

Ordinary `registration_only` runs execute CPU audits and write a clearly labelled public-test filename placeholder, without model inference. Formal `KAGGLE_IS_COMPETITION_RERUN` ignores registration_only, requires actualL4×4, runs the real model, and atomically updates hidden predictions. A placeholder is not a model score.

L4 metadata Quick Save v21 was saved with `--no-run`; it did not start an ordinary GPU session. Quick Save-only v18 could not submit because no submission.json existed. The competition requires an output FileName. Explicit completed CPU v20 was accepted as official submission **57648699**, with v21 L4 metadata current. Hardware inheritance and score are still pending; acceptance does not establish GPU execution. Previous CPU integration submission57645178 failed with a generic unhandled error and no score. Submission downloads were denied by the service; no restriction was bypassed.

## Strict GPU accounting

The user authorised less than2 hours total GPU notebook validation, then non-GPU ordinary pushes/runs. All4 complete notebook logs sum5707.410s =95.1235min. Conservative charge rounds each complete duration up to a minute and adds120s/run:6360s =106min, below the unchanged6900s self-cap and7200s user cap. The ledger is closed for further ordinary GPU inference requests. Historic requested timeouts9900s remain recorded separately; unused finished timeouts were released only after complete logs existed. The pre-rebase ledger is preserved.

Use scripts/push_kaggle.py and the shared ledger. Default execution is CPU. `--submission-snapshot` is mutually exclusive with `--gpu` and always uses `--no-run`; it changes metadata only. All ordinary executions after G04 were CPU. Official competition reruns are tracked separately as the user-authorised leaderboard evidence path.

## Reproduce the core

Linux, libseccomp.so.2, Python, NumPy2.3.5. Core requires no Torch.

```bash
cd code
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m arcsolver.cli \
 --tasks ../references/kaggle-public-evaluation/arc-agi_evaluation_challenges.json \
 --labels ../references/kaggle-public-evaluation/arc-agi_evaluation_solutions.json \
 --config configs/A1-one-program.json --subset 4 --budget 2000 \
 --url http://127.0.0.1:8000/v1 --gpu-count 4 --out ../experiments/reproduction
```

This CLI assumes a separately available local model server; it does not start one or bypass the GPU ledger. Use --subset 0 for all120 tasks. Optional --concurrency 2 requires adequate server slots. Production defaults should not be promoted from CPU fixtures.

```bash
python scripts/build_notebook.py --target flash --seconds 3000 --subset 4 \
 --baseline-only --stream --thinking-budget 4096 --compiled-serving \
 --serving-seqs 2 --task-concurrency 2 --registration-only \
 --slug arc2-flash-program-induction
```

Generated notebook embeds solver source, tests/configs and its source hash. Match the actual saved notebook/version before interpreting results. `experiments/G04-source` is the immutable executed eager baseline; current code contains subsequent CPU-tested fixes and experimental controls.

## Data revision and metrics

Actual Kaggle public evaluation has120 tasks/172 test outputs and differs from the original frozen official repo in6 task contents and5 labels. Both versions and original licences remain preserved. Actual challenge SHA256 `e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8`; solutions SHA256 `84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1`.

Fixed4 IDs:eee78d87,bf45cf4b,d35bdbdc,aa4ec2a5. Seed20261024, stratified using training shape/size only. Kaggle4/12/120 manifests and fixed4 data are archived. Test labels never enter model/search; evaluators consume them only after selecting predictions. Posthoc analyzer rejects mismatched input digests/test counts.

Report per-output exact pass@1/pass@2, scheduled/completed denominators, exact-train-fit, conditional test fit, verified correct tasks/outputs per GPU-device-hour, candidate/proposal/archive/duplicate counts, repairs, failures, runtime/VRAM. An entire-task verified solution requires one train-fitting program to match all its test outputs. GPU-device-hours mean elapsed time×attached devices, not Kaggle billing units.

```bash
python scripts/analyze_evaluation.py --run ../experiments/G04-output/arc2-results \
 --tasks ../references/kaggle-public-evaluation/arc-agi_evaluation_challenges.json \
 --labels ../references/kaggle-public-evaluation/arc-agi_evaluation_solutions.json \
 --out ../experiments/G04-analysis.json
python scripts/audit_traces.py --traces ../experiments/G04-output/arc2-results/A1/traces \
 --out ../experiments/G04-rawcall-audit.json
```

`ablate_archives.py` compares filtering/selection over an already-generated pool. It does not establish equal-generation-budget effects. G04 has no ARC program pool, so no search/repair/diversity gain can be inferred. Duck mode is bounded generic Python exploration in fresh processes, not an exact original persistent Duck reproduction.

## Correctness and trace safeguards

Fresh mandatory seccomp process per input grid; no filesystem/network/process creation or credential inheritance after library preload. Memory/CPU/wall/stdout bounds apply. Missing isolation fails closed. Parent validates raw worker JSON, result count and integer colour-grid domain. Module state cannot leak across grid executions.

Exact verification never substitutes cell similarity. All candidate lineage and counterexamples are archived. Final Python parsing excludes thinking-only code. Optional SSE retains partial reasoning and final content separately on deadlines; no accuracy gain is claimed. Exact tokenizer context counting preserves demonstrations while fitting output budget. Model calls reserve verification time. Even failed generation calls now archive their prompt/budget with the hidden-label firewall intact.

Atomic checkpoint writes use unique temporary files; a reproduced four-writer race in the previous shared-temp implementation is retained as evidence. Concurrent task results merge on the evaluator thread and persist submission after every result.

## Offline runtime and provenance

Flash attachments: dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/transformers/default/1; foysalemonshanto/vllm-0-30-0-runtime. 27B attachments: rahim3/qwen3-8-27b-bf16; saltb0x/arc3-vllm-wheelhouse-v0271-cu129. Both attach liangwanyiudavid/arc2-runtime-builder for hash-checked standalone CPython3.12.15. Current Kaggle host is3.13. Flash requires Qwen4Exp support in pinned vLLM0.30, not0.27.1.

New code MIT-0. Upstream licences/URLs/hashes are in references/SOURCE_CATALOG.json and SOURCE_MANIFEST.json. Downloaded papers, source snapshots, actual benchmark data, prompts/configs, source notebooks, raw traces, metrics and interpreter bytes are included. Large model/runtime attachments remain version/manifest-pinned external Kaggle inputs rather than duplicated weights. One unused interrupted interpreter download is separately preserved with a manifest/Drive link; canonical complete interpreter remains included. Credentials are excluded.

```bash
python code/scripts/package_checkpoint.py --root . --name ARC2_Build_Checkpoint_06
python code/scripts/verify_checkpoint.py packages/ARC2_Build_Checkpoint_06.zip
```

Next evidence needed: official rerun outcome/hardware, compiled throughput and final-code yield, paired runtime/budget ablations, model repair/search/diversity ablations, full120 public evaluation, and a measured high-scoring submission. These remain open research items.
