# Nemotron VoiceChat P4-3 local validation

Issue: https://github.com/sgl-project/sglang-omni/issues/2459 (P4-3).

Date: 2026-10-18 (host log date). Worktree: `/root/autodl-tmp/wt-p43`; branch: `verify/nemotron-rvq-batch`.

Baseline: P4-4 `f19bef13`; initial P4-3: `0cba5237`. This comparison isolates batching on top of the existing schedule and graph changes.

## Implementation

Compatible Talker rows use the runner-wide sampling settings, FP32 RVQ and MoG head, and one `generate_codes` call. The scheduler request order determines the hidden rows and output scatter. Graph buckets cover the configured maximum running requests. Graph outputs are cloned before later replay; all codes cross to CPU once at the result boundary, with per-request slices retained by their owners.

Sampling uses the global PyTorch RNG. A fixed seed and identical batch schedule reproduce outputs; request order, batch composition, bucket padding and numerical changes can change generated codes. It does not promise per-request seed invariance across batch changes.

## Tests

- Initial CPU/CUDA RVQ suite: 38 passed.
- Extended CPU/CUDA RVQ suite: 40 passed, including request-coupled Gumbel/Gaussian draws under permutation, shrinking batches and single-row execution.
- Nemotron CPU suite: 55 passed, 22 skipped (GPU/checkpoint gated tests).
- Runner-specific CPU tests: 4 passed; eager/graph dispatch, reorder/removal, previous codes, retained history/chunks, waiting and terminating requests.
- Black, isort, Ruff, leading-underscore checker, if/else checker and `git diff --check` passed on changed code. Ruff reports an existing custom noqa spelling warning.

## Real-weight component benchmark

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch 2.13.0+cu130. Local 11B checkpoint, original FP32 RVQ/MoG weights, the first eight captured real hidden states. Three warmups, 30 timed iterations per setting; host wall time includes CPU output copy. Serial variants copy each row, batched variants copy the whole result once. The serial graph path invokes the same B=1 graph once per request, reproducing the P4-4 execution pattern. This measures RVQ only, not complete inference.

| Batch | Serial graph ms | Batched graph ms | Speedup | Serial eager ms | Batched eager ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.857 | 2.851 | 1.00x | 6.889 | 6.794 |
| 2 | 5.707 | 3.168 | 1.80x | 13.199 | 6.623 |
| 3 | 8.545 | 3.299 | 2.59x | 19.794 | 6.592 |
| 4 | 11.381 | 3.294 | 3.45x | 26.111 | 6.507 |
| 8 | 22.771 | 3.282 | 6.94x | 52.214 | 6.630 |

Graph capture for buckets 1/2/4/8: 0.512 s; peak allocated component memory: 0.879 GiB.

Real weights: padded eager and graph outputs matched exactly for nonmonotonic batch sequence 1/3/2/8/4/1/7/2. Every tested batch refreshed noise, reproduced with the same seed, and retained previous output after another replay. Repeated identical-input sampling produced distinct outputs at batches 1/3/8; this is a freshness check, not a statistical quality guarantee.

## End-to-end validation

One fresh process per variant, one full single-request warmup, then three distinct checkpoint audio clips submitted concurrently. Both Thinker and Talker allow four running requests. The configurations use Thinker memory fraction 0.4, Talker KV cache 4 GiB, Triton Talker attention and Talker seed 1234. RVQ graphs remain enabled; scheduler logs reporting `cuda graph: False` refer to the backbone. Actual Talker batch sizes are recorded separately. The single-request warmup includes first-use compilation and is excluded from performance conclusions. One concurrent wave per variant is an exploratory comparison, not a confidence interval or a broad performance guarantee.

Both variants completed all four requests (one warmup plus three concurrent), with observed Talker batch sizes 1/2/3. All four paired texts and audio sample counts matched; all four PCM hashes differed. The global RNG and extra graph-bucket warmups do not preserve baseline per-request random streams.

| Concurrent clip | Baseline latency s | Candidate latency s | Audio duration s |
| --- | ---: | ---: | ---: |
| interruptions | 41.144 | 36.282 | 30.08 |
| turn_taking | 44.013 | 39.271 | 41.12 |
| tool_call | 70.911 | 67.573 | 84.48 |

Total concurrent wave: **70.932 s → 67.592 s**, a **4.71% latency reduction** (1.049x throughput for this fixed set). This single pair can include scheduling and first-use shape compilation differences. It is consistent with the RVQ savings but does not isolate their full end-to-end effect.

Warmup latency was 53.751 s versus 27.553 s; this includes unequal first-use compilation/cache costs and is **not** evidence of a 1.95x batch-1 speedup.

Quality scoring used the cached Whisper large-v3-turbo ASR model with English normalization, comparing ASR text against the Thinker response, plus cached UTMOS. All four outputs per variant were scored.

| Request | Baseline WER | Candidate WER | Baseline UTMOS | Candidate UTMOS |
| --- | ---: | ---: | ---: | ---: |
| sequential: turn_taking | 0.00% | 0.00% | 4.447 | 4.442 |
| concurrent: interruptions | 65.99% | 68.02% | 4.052 | 4.459 |
| concurrent: turn_taking | 0.00% | 3.28% | 4.438 | 4.414 |
| concurrent: tool_call | 5.68% | 2.27% | 4.372 | 4.401 |

Across all four requests, corpus WER was **33.17% → 33.91%** and mean UTMOS was **4.327 → 4.429**. WER therefore slightly worsened while UTMOS improved; this small sample does not prove quality non-regression.

The interruptions clip dominates corpus WER in both variants (65.99% / 68.02%). Inspection shows that the full Thinker text retains long continuations while both ASR transcripts contain short prefixes of the interrupted turns. This makes full-response WER sensitive to interruption/truncation behavior; it cannot be read purely as an acoustic-quality score. Excluding that clip, WER over the remaining three recordings was 2.38% → 1.90%; the full per-clip results above are retained to avoid hiding the worse metric.

Functional and stochastic checks passed: correct row ownership and output lifetimes, graph/eager agreement with matched RNG shapes, fresh replay draws and complete real-model inference. Remaining uncertainty is broad statistical output-quality equivalence and repeatable end-to-end gains across more seeds/workloads.

## Local reproduction and artifacts

All logs, scripts, JSON and WAV outputs are under `/root/autodl-tmp/eval/p43/`. `REPRODUCE.sh` contains the full local procedure, including the two process launches and ASR/UTMOS scoring. It does not shut down the machine.

```bash
source /root/autodl-tmp/omni-env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd /root/autodl-tmp/wt-p43
python -m pytest tests/unit_test/nemotron_voicechat/test_talker_rvq.py -q
CUDA_VISIBLE_DEVICES="" python -m pytest tests/unit_test/nemotron_voicechat -q
python /root/autodl-tmp/eval/p43/component_batch.py
```
