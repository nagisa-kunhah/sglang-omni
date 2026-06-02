# MOSS-TTS Reference Encoder Compile Experiment

## Motivation

#637

## Modifications

Code files touched for the experiment:

- `sglang_omni/models/moss_tts/reference_audio_encoder.py`
  - Added/configured `MossReferenceAudioEncoder` compile wrapping for the real
    upstream reference-audio tokenizer path.
  - Compile target is configurable and restores the original method on failure.

- `sglang_omni/models/moss_tts/config.py`
  - Added preprocessing compile-related config knobs.
  - Kept encoder compile disabled by default.
  - Set preprocessing reference encoder dtype to `float32` for the tokenizer
    path because upstream quantizer code contains `.float()` conversions.

- `sglang_omni/models/moss_tts/stages.py`
  - Wires preprocessing compile options into `MossReferenceAudioEncoder`.

- `tests/unit_test/moss_tts/test_pipeline.py`
  - Adds/updates unit coverage for compile argument wiring, warmup behavior,
    failure restoration, dtype/default config, and preprocessing executor setup.

Benchmark/repro files added:

- `benchmarks/eval/benchmark_moss_tts_reference_encoder_compile.py`
  - Isolated paired benchmark for:
    `processor.encode_audios_from_wav -> batch_encode -> _encode_frame`.
  - Supports unmeasured `--prewarm-requests`.

- `benchmarks/eval/benchmark_moss_tts_encode_frame_compile.py`
  - Rough, non-official service benchmark helper for paired compile-on/off
    `/v1/audio/speech` requests.
  - Starts each server in a process group and uses separate ports for paired
    cases.

- `benchmarks/eval/moss_tts_compile_server_configs/`
  - Checked-in YAML configs for maintainers to reproduce server behavior:
    - `compile_off.yaml`
    - `compile_batch_encode_default.yaml`
    - `compile_batch_encode_reduce_overhead.yaml`
    - `compile_batch_encode_max_no_cg.yaml`
    - `compile_batch_encode_fullgraph.yaml`

## Optimization Targets Tried

Tried compile targets:

- `audio_tokenizer._encode_frame`
- `audio_tokenizer.batch_encode`

Tried compile modes:

- `default`
- `reduce-overhead`
- `max-autotune-no-cudagraphs`

Tried fullgraph settings:

- `fullgraph=True`
- `fullgraph=False`

Important observations:

- `fullgraph=True` fails on upstream `types.UnionType` / `typing.cast(...)`.
- `reduce-overhead` can fail in the service worker path with an Inductor
  cudagraph TLS `AssertionError`.
- `max-autotune-no-cudagraphs` can return a valid service response, but the
  isolated paired benchmark did not show a latency win.
- Non-fullgraph service logs showed Dynamo recompilation pressure, including
  `_sa_block` hitting `config.recompile_limit`.

## Reproduction Commands

Common environment:

```bash
cd /root/sglang-omni
export PYTHONPATH=/root/sglang-omni
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/hy-tmp/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export SGLANG_OMNI_STARTUP_TIMEOUT=900
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Generate a request payload:

```bash
python -m benchmarks.eval.benchmark_moss_tts_encode_frame_compile \
  --model-path /tmp/moss-tts-v15 \
  --ref-audio /tmp/moss_ref_3s.wav \
  --output-dir /tmp/moss_tts_request_payload \
  --prepare-only
```

Start a server with one checked-in config:

```bash
sgl-omni serve \
  --config benchmarks/eval/moss_tts_compile_server_configs/compile_batch_encode_max_no_cg.yaml \
  --host 127.0.0.1 \
  --port 18103 \
  --log-level info
```

Send one request:

```bash
curl -sS \
  -X POST http://127.0.0.1:18103/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/moss_tts_request_payload/request.json \
  --output /tmp/moss_tts_out.wav \
  -w '\nstatus=%{http_code} time=%{time_total} size=%{size_download} type=%{content_type}\n'
```

Run the isolated reference encoder benchmark:

```bash
python -m benchmarks.eval.benchmark_moss_tts_reference_encoder_compile \
  --model-path /tmp/moss-tts-v15 \
  --ref-audio /tmp/moss_ref_3s.wav \
  --device cuda:0 \
  --encoder-dtype float32 \
  --compile-target batch_encode \
  --compile-mode reduce-overhead \
  --prewarm-requests 10 \
  --cold-requests 1 \
  --warm-requests 20 \
  --output-json /tmp/moss_tts_batch_encode_bench_reduce_overhead_prewarm10.json
```

Run the rough paired service helper:

```bash
python -m benchmarks.eval.benchmark_moss_tts_encode_frame_compile \
  --model-path /tmp/moss-tts-v15 \
  --ref-audio /tmp/moss_ref_3s.wav \
  --compile-target batch_encode \
  --compile-mode max-autotune-no-cudagraphs \
  --encoder-dtype float32 \
  --cold-requests 1 \
  --warm-requests 2 \
  --port 18220 \
  --output-dir /tmp/moss_tts_pair_max_no_cg
```

For graph break / recompile diagnosis:

```bash
TORCH_LOGS="graph_breaks,recompiles" \
TORCHDYNAMO_VERBOSE=1 \
sgl-omni serve \
  --config benchmarks/eval/moss_tts_compile_server_configs/compile_batch_encode_max_no_cg.yaml \
  --host 127.0.0.1 \
  --port 18103 \
  --log-level info
```

## Benchmark & Profiling

Isolated `batch_encode`, `mode=reduce-overhead`, `prewarm=10`, `warm=20`:

```text
compile_on_warm_mean_s:  0.061749
compile_off_warm_mean_s: 0.056812
warm_speedup:            0.9200
```

Rough service run, `batch_encode`, `mode=max-autotune-no-cudagraphs`:

```text
compile_on cold#0: status=200, latency_s=10.290, size=57644, type=audio/wav
compile_on warm#1: status=200, latency_s=0.695,  size=57644, type=audio/wav
```

The rough service run did not collect a valid paired compile-off result in the
same run. It should not be treated as an official benchmark.

## Initial Conclusion

No speedup was observed in the isolated paired benchmark. Keep encoder compile
disabled by default.

Recommended next directions:

- Prefer reference audio token caching for repeated voices/reference audio.
- If compile is still investigated, try compiling smaller stable encoder
  submodules instead of compiling all of `batch_encode` or `_encode_frame`.
- Avoid `reduce-overhead` in the service worker path unless the cudagraph TLS
  failure is resolved.

## Checklist

- [ ] Format code according to pre-commit.
- [x] Add unit tests.
- [x] Add maintainer repro configs and commands.
- [x] Provide initial benchmark/profiling results.
- [ ] Add accuracy test.

