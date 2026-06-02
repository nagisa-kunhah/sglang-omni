# MOSS-TTS Reference Encoder Compile Experiment

## Reason

The original opt-in compile path in this workspace could not run the real
MOSS-TTS preprocessing path with the configured `encoder_dtype="bfloat16"`.
Both eager and compile attempts failed when reference audio reached the audio
tokenizer:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

The failure happened because the upstream processor builds float32 waveform
tensors and then calls `audio_tokenizer.batch_encode(...)`, while the stage
factory moves the tokenizer weights to bfloat16. The existing compile helper
also wrapped `audio_tokenizer.encode`, but the real upstream path used by
`encode_audios_from_wav` prefers `batch_encode -> _encode_frame`, so the
compiled callable was not the actual preprocessing hot path.

## Method

The experiment uses a local `MossReferenceAudioEncoder` wrapper for opt-in
reference audio preprocessing:

- Keep file path/data URI decoding, mono mixdown, resampling, loudness
  normalization, padding, and CPU output formatting outside the compiled graph.
- Build padded `input_values` using the audio tokenizer parameter dtype, so
  bfloat16 tokenizer weights receive bfloat16 inputs.
- Run the tokenizer encode call under CUDA autocast when the tokenizer weights
  are float16/bfloat16. The tokenizer quantizer contains internal `.float()`
  casts before convolution layers, so dtype-correct padded inputs alone are not
  sufficient for the configured bfloat16 path.
- Compile only `_encode_padded_eager(...)`, which calls:

```python
audio_tokenizer.encode(
    input_values,
    padding_mask=padding_mask,
    num_quantizers=n_vq,
    return_dict=True,
)
```

- Preserve the processor-facing reference contract as `list[torch.Tensor]`,
  where each tensor has shape `[T, NQ]`, dtype `torch.long`, and CPU device.
- Keep raw tensor references compatible by passing them through as `[tensor]`.
- Only instantiate the wrapper when `enable_encoder_torch_compile=True`; the
  disabled path keeps the upstream processor behavior.
- When compile is enabled, patch the tokenizer attention class at runtime to
  remove two `typing.cast(MHAState | None, ...)` calls. These casts are runtime
  no-ops, but PyTorch Dynamo cannot fullgraph trace the `types.UnionType`
  expression in the downloaded tokenizer source.
- Apply the same runtime-only treatment to `_encode_frame`, where another
  `typing.cast(A | B, ...)` around `self.quantizer` creates the same Dynamo
  fullgraph failure.

The full MOSS unit file also exposed that `MossTTSModelRunner.custom_prefill_forward`
only staged `forward_batch.input_embeds` and returned `None`. The scheduler/test
contract expects it to initialize attention metadata, call the model with
`input_embeds`, disable cuda graph for this custom path, and return the model
result. That behavior was restored so the broader MOSS file can run beyond the
preprocessing tests.

During the real server smoke test, the first compatible implementation passed
`mrope_positions` into `MossTTSDelaySGLangModel.forward(...)`. That argument is
valid for several Qwen-family model runners in this repository, but the MOSS
forward signature is:

```python
forward(input_ids, positions, forward_batch, input_embeds=None, ...)
```

The runner now passes `forward_batch=forward_batch` and does not pass
`mrope_positions`, matching the concrete MOSS model contract used by the loaded
checkpoint.

The next server smoke reached scheduler result handling and failed because the
custom prefill path returned the bare `LogitsProcessorOutput` from the model.
The omni scheduler expects either `None` or a `GenerationBatchResult` with
`logits_output`, `next_token_ids`, and cuda-graph metadata. MOSS now follows the
same pattern as the Qwen3-TTS runner: wrap the model output in
`GenerationBatchResult(logits_output=..., can_run_cuda_graph=False)` so the base
runner can perform sampling and post-prefill processing normally.

## Benchmark Inputs

Weights are downloaded under `/hy-tmp/huggingface/hub`:

- `OpenMOSS-Team/MOSS-TTS-v1.5`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer`

The reference audio for timing is a 3 second clip derived from the local demo
audio:

```text
/tmp/moss_ref_3s.wav
```

For the full server comparison, `sglang_omni/models/moss_tts/config.py` was
temporarily toggled between `enable_encoder_torch_compile=True` and
`enable_encoder_torch_compile=False` while keeping the same local model
snapshot, request payload, reference wav, and seed. The toggle is intentionally
kept explicit in config for this experiment so each server process has a single
unambiguous preprocessing mode.

The upstream no-wrapper/no-compile path could not be used as an e2e baseline in
this environment. It calls the downloaded processor audio loader, which imports
`torchcodec`; even after installing Ubuntu FFmpeg 4 shared libraries, the local
`torchcodec` binary is ABI-incompatible with `torch 2.9.1+cu128`:

```text
OSError: .../torchcodec/libtorchcodec_core4.so: undefined symbol: torch_from_blob
```

To keep the comparison e2e and isolate compile itself, preprocessing now has a
separate `enable_reference_audio_encoder` flag. With
`enable_reference_audio_encoder=True` and `enable_encoder_torch_compile=False`,
the server uses the same real local wrapper, audio decoding, dtype handling, and
request path as the compile run, but executes `_encode_padded_eager` without
`torch.compile`.

## Results

### Local Weights

- MOSS-TTS v1.5 cache:
  `/hy-tmp/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-v1.5` (`16G`)
- MOSS Audio Tokenizer cache:
  `/hy-tmp/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer` (`6.7G`)
- These two model caches total about `22.7G`.
- The full Hugging Face hub cache under `/hy-tmp/huggingface/hub` was `54G`,
  including unrelated cached artifacts.

### Preprocessing Encoder Microbenchmark

The benchmark used the same real 3 second reference wav, local MOSS audio
tokenizer weights, bfloat16 tokenizer dtype, and no mocks.

- Eager wrapper times:
  `[0.745684, 0.183624, 0.296294, 0.300248, 0.206384, 0.295578, 0.304564, 0.295825]`
- Eager steady-state (`times[2:]`) mean: `0.283149s`
- Eager steady-state median: `0.296060s`
- Compiled wrapper times:
  `[121.563376, 0.154634, 0.289629, 0.294582, 0.209799, 0.297749, 0.296454, 0.211642]`
- Compiled first request includes compile cost: `121.563376s`
- Compiled steady-state (`times[2:]`) mean: `0.266643s`
- Compiled steady-state median: `0.292106s`

Preprocessing-only compile effect:

- Mean improvement: about `5.8%`
- Median improvement: about `1.3%`
- The first compile cost is large enough that this path needs warmup or enough
  request volume to amortize it.

### Full Server E2E

Request payload:

```json
{
  "model": "/tmp/moss-tts-v15",
  "input": "hello world",
  "voice": "default",
  "ref_audio": "/tmp/moss_ref_3s.wav",
  "ref_text": "water",
  "seed": 1234
}
```

Compile-on service mode:

```text
enable_reference_audio_encoder=True
enable_encoder_torch_compile=True
encoder_torch_compile_mode=max-autotune-no-cudagraphs
```

Compile-on results:

- First successful real e2e request: `200 audio/wav`, `49964 bytes`,
  `17.557s`
- Output validation: RIFF/WAVE PCM 16-bit mono, 24000 Hz, duration about
  `1.040s`
- Fixed-seed warm requests:
  `[0.665, 0.663, 0.635, 0.642, 0.632]`
- Output size for all fixed-seed warm requests: `57644 bytes`
- Warm mean: `0.6474s`
- Warm median: `0.642s`

Eager-wrapper service mode:

```text
enable_reference_audio_encoder=True
enable_encoder_torch_compile=False
```

Eager-wrapper results:

- First successful real e2e request: `200 audio/wav`, `57644 bytes`,
  `2.423s`
- Fixed-seed warm requests:
  `[0.718, 0.697, 0.716, 0.700, 0.706]`
- Output size for all fixed-seed warm requests: `57644 bytes`
- Warm mean: `0.7074s`
- Warm median: `0.706s`

Full e2e compile effect under this fixed-seed wrapper-vs-wrapper comparison:

- Warm mean improvement: about `8.5%`
- Warm median improvement: about `9.1%`
- The compile-on cold path is much slower because of compilation. The
  comparable steady-state result is only after the compiled graph has been
  created.
