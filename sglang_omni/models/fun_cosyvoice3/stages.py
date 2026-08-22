# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import torch

from sglang_omni.models.fun_cosyvoice3.flow_batch import (
    FlowBatchInput,
    infer_flow_batch,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec

# This is an admission budget, not a maximum supported request length. The
# scheduler admits a request that exceeds it as a singleton Flow batch and
# defers following requests to the next batch.
_DEFAULT_FLOW_BATCH_ADMISSION_FRAMES = 2000

_COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)


def load_state(payload: StagePayload) -> FunCosyVoice3State:
    return _load_pipeline_state(payload, FunCosyVoice3State)


def store_state(payload: StagePayload, state: FunCosyVoice3State) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
) -> tuple[Any, Any]:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(_COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    del cv.model.llm
    return flow, hift


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.engine_builder import (
        FunCosyVoice3EngineBuilder,
    )

    return FunCosyVoice3EngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


@dataclass(frozen=True)
class _PreparedFlowRequest:
    index: int
    sample_rate: int
    flow_input: FlowBatchInput


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        fp16: bool = False,
        flow_batch_bucket_frames: int = 50,
    ) -> None:
        if flow_batch_bucket_frames <= 0:
            raise ValueError("flow_batch_bucket_frames must be greater than zero")
        if not isinstance(flow.decoder.estimator, torch.nn.Module):
            raise RuntimeError(
                "Fun-CosyVoice3 requires the PyTorch Flow estimator from the pinned "
                "CosyVoice commit; TensorRT Flow is not supported"
            )
        self._flow = flow
        self._hift = hift
        self._fp16 = fp16
        self._flow_batch_bucket_frames = flow_batch_bucket_frames

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        # The AR runner stores one-element tensors per step, which serialize as
        # ``[num_tokens, 1]``. Flow consumes one unbatched token sequence here.
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        prepared = [
            _PreparedFlowRequest(
                index=index,
                sample_rate=state.sample_rate,
                flow_input=self._make_flow_input(state, codes),
            )
            for index, (state, codes) in enumerate(items)
        ]
        results: list[tuple[Any, int] | None] = [None] * len(prepared)
        buckets: dict[int, list[_PreparedFlowRequest]] = defaultdict(list)
        for request in prepared:
            buckets[self._flow_bucket_key(request.flow_input)].append(request)

        for bucket in buckets.values():
            with torch.autocast(
                device_type=current_platform.device_type, enabled=self._fp16
            ):
                mel_list = infer_flow_batch(
                    self._flow,
                    [request.flow_input for request in bucket],
                )
            for request, mel in zip(bucket, mel_list, strict=True):
                results[request.index] = (
                    self._mel2wav(mel),
                    request.sample_rate,
                )

        if any(result is None for result in results):
            raise RuntimeError("Fun-CosyVoice3 vocoder did not decode every request")
        return [cast(tuple[Any, int], result) for result in results]

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.decode_payloads([payload])
        if len(results) != 1:
            raise RuntimeError(
                f"Fun-CosyVoice3 vocoder returned {len(results)} results for 1 input"
            )
        return results[0]

    def _make_flow_input(
        self,
        state: FunCosyVoice3State,
        codes: torch.Tensor,
    ) -> FlowBatchInput:
        prompt_token = (
            torch.as_tensor(state.flow_prompt_speech_token, dtype=torch.int32).reshape(
                1, -1
            )
            if state.flow_prompt_speech_token is not None
            else torch.zeros(1, 0, dtype=torch.int32)
        )
        prompt_feat = (
            torch.as_tensor(state.flow_prompt_speech_feat).reshape(1, -1, 80)
            if state.flow_prompt_speech_feat is not None
            else torch.zeros(1, 0, 80)
        )
        embedding = (
            torch.as_tensor(state.flow_embedding).reshape(1, -1)
            if state.flow_embedding is not None
            else torch.zeros(1, 192)
        )
        return FlowBatchInput(
            token=codes.reshape(1, -1).to(torch.int32),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def _flow_bucket_key(self, item: FlowBatchInput) -> int:
        total_mel = self._flow_total_mel_frames(item)
        return (
            total_mel + self._flow_batch_bucket_frames - 1
        ) // self._flow_batch_bucket_frames

    def _flow_total_mel_frames(self, item: FlowBatchInput) -> int:
        total_tokens = item.prompt_token.shape[1] + item.token.shape[1]
        return total_tokens * self._flow.token_mel_ratio

    def _flow_scheduler_cost(self, payload: StagePayload) -> int:
        state, codes = self.prepare_item(payload)
        total_mel = self._flow_total_mel_frames(self._make_flow_input(state, codes))
        return (
            (total_mel + self._flow_batch_bucket_frames - 1)
            // self._flow_batch_bucket_frames
            * self._flow_batch_bucket_frames
        )

    def _mel2wav(self, tts_mel: torch.Tensor) -> torch.Tensor:
        tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech.detach().cpu()

    def store_result(
        self,
        payload: StagePayload,
        state: FunCosyVoice3State,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Fun-CosyVoice3 vocoder did not return audio")
        audio_payload = audio_waveform_payload(wav, source_hint="Fun-CosyVoice3")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    flow_batch_bucket_frames: int = 50,
    flow_batch_admission_frames: int = _DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
) -> SimpleScheduler:
    if flow_batch_admission_frames <= 0:
        raise ValueError("flow_batch_admission_frames must be greater than zero")
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
    )

    vocoder = _CosyVoice3Vocoder(
        flow,
        hift,
        fp16=(dtype == "float16"),
        flow_batch_bucket_frames=flow_batch_bucket_frames,
    )

    return SimpleScheduler(
        vocoder.decode_payload,
        batch_compute_fn=vocoder.decode_payloads,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder._flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
    )
