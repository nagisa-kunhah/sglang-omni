# MOSS-TTS Reference Encoder Compile Server Configs

These configs are small, reproducible server configs for maintainers to compare
MOSS-TTS reference-audio preprocessing compile modes. They are not official
benchmarks.

Default local paths:

```bash
MODEL=/tmp/moss-tts-v15
REF_AUDIO=/tmp/moss_ref_3s.wav
```

If the model path differs, edit `model_path` and `name` in the YAML before
starting the server.

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

Generate a request payload with a data-URI reference audio:

```bash
python -m benchmarks.eval.benchmark_moss_tts_encode_frame_compile \
  --model-path /tmp/moss-tts-v15 \
  --ref-audio /tmp/moss_ref_3s.wav \
  --output-dir /tmp/moss_tts_request_payload \
  --prepare-only
```

Start one server at a time:

```bash
sgl-omni serve \
  --config benchmarks/eval/moss_tts_compile_server_configs/compile_off.yaml \
  --host 127.0.0.1 \
  --port 18100 \
  --log-level info
```

Send one request:

```bash
curl -sS \
  -X POST http://127.0.0.1:18100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/moss_tts_request_payload/request.json \
  --output /tmp/moss_tts_out.wav \
  -w '\nstatus=%{http_code} time=%{time_total} size=%{size_download} type=%{content_type}\n'
```

Configs:

```text
compile_off.yaml
  Baseline. Reference audio encoder compile disabled.

compile_batch_encode_default.yaml
  Compile audio_tokenizer.batch_encode with torch.compile mode=default.

compile_batch_encode_reduce_overhead.yaml
  Compile audio_tokenizer.batch_encode with mode=reduce-overhead.
  In the rough service run this mode returned 500 due an Inductor cudagraph TLS
  AssertionError in the worker path.

compile_batch_encode_max_no_cg.yaml
  Compile audio_tokenizer.batch_encode with mode=max-autotune-no-cudagraphs.
  In the rough service run this mode returned a valid audio/wav response.

compile_batch_encode_fullgraph.yaml
  Compile audio_tokenizer.batch_encode with fullgraph=True and mode=default.
  This is expected to fail on the upstream types.UnionType / typing.cast path.
```

For graph-break and recompile diagnosis:

```bash
TORCH_LOGS="graph_breaks,recompiles" \
TORCHDYNAMO_VERBOSE=1 \
sgl-omni serve \
  --config benchmarks/eval/moss_tts_compile_server_configs/compile_batch_encode_max_no_cg.yaml \
  --host 127.0.0.1 \
  --port 18103 \
  --log-level info
```

