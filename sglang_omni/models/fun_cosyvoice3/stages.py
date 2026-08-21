# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import torch

from sglang_omni.models.fun_cosyvoice3.flow_batch import (
    FlowBatchInput,
    flow_batch_unsupported_reason,
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

logger = logging.getLogger(__name__)

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
        enable_flow_batch: bool = True,
        flow_batch_bucket_frames: int = 50,
    ) -> None:
        if flow_batch_bucket_frames <= 0:
            raise ValueError("flow_batch_bucket_frames must be greater than zero")
        self._flow = flow
        self._hift = hift
        self._fp16 = fp16
        self._enable_flow_batch = enable_flow_batch
        self._flow_batch_bucket_frames = flow_batch_bucket_frames
        self._flow_batch_unsupported_reason = flow_batch_unsupported_reason(flow)
        if enable_flow_batch and self._flow_batch_unsupported_reason is not None:
            logger.warning(
                "Fun-CosyVoice3 Flow batch is unavailable; falling back to upstream "
                "serial inference: %s",
                self._flow_batch_unsupported_reason,
            )
        logger.info(
            "Fun-CosyVoice3 Flow batch configuration: requested=%s, supported=%s, "
            "bucket_frames=%d",
            enable_flow_batch,
            self._flow_batch_unsupported_reason is None,
            flow_batch_bucket_frames,
        )

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
        serial_requests: list[_PreparedFlowRequest] = []
        buckets: dict[int, list[_PreparedFlowRequest]] = defaultdict(list)
        for request in prepared:
            if self._is_flow_batch_eligible(request.flow_input):
                buckets[self._flow_bucket_key(request.flow_input)].append(request)
            else:
                serial_requests.append(request)

        for request in serial_requests:
            item = request.flow_input
            mel = self._token2mel(
                item.token, item.prompt_token, item.prompt_feat, item.embedding
            )
            results[request.index] = (
                self._mel2wav(mel),
                request.sample_rate,
            )

        for bucket in buckets.values():
            if len(bucket) == 1:
                item = bucket[0].flow_input
                mel_list = [
                    self._token2mel(
                        item.token,
                        item.prompt_token,
                        item.prompt_feat,
                        item.embedding,
                    )
                ]
            else:
                with torch.autocast(
                    device_type=current_platform.device_type, enabled=self._fp16
                ):
                    mel_list = infer_flow_batch(
                        self._flow, [request.flow_input for request in bucket]
                    )
            for request, mel in zip(bucket, mel_list, strict=True):
                results[request.index] = (
                    self._mel2wav(mel),
                    request.sample_rate,
                )

        if any(result is None for result in results):
            raise RuntimeError("Fun-CosyVoice3 vocoder did not decode every request")
        return [cast(tuple[Any, int], result) for result in results]

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

    def _is_flow_batch_eligible(self, item: FlowBatchInput) -> bool:
        if (
            not self._enable_flow_batch
            or self._flow_batch_unsupported_reason is not None
        ):
            return False
        if item.token.ndim != 2 or item.token.shape[0] != 1 or item.token.shape[1] <= 0:
            return False
        if item.prompt_token.ndim != 2 or item.prompt_token.shape[0] != 1:
            return False
        if (
            item.prompt_feat.ndim != 3
            or item.prompt_feat.shape[0] != 1
            or item.prompt_feat.shape[2] != self._flow.output_size
        ):
            return False
        if (
            item.prompt_feat.shape[1]
            != item.prompt_token.shape[1] * self._flow.token_mel_ratio
        ):
            return False
        if item.embedding.ndim != 2 or item.embedding.shape[0] != 1:
            return False
        expected_embedding_size = getattr(
            self._flow.spk_embed_affine_layer, "in_features", None
        )
        return (
            expected_embedding_size is None
            or item.embedding.shape[1] == expected_embedding_size
        )

    def _flow_bucket_key(self, item: FlowBatchInput) -> int:
        total_tokens = item.prompt_token.shape[1] + item.token.shape[1]
        total_mel = total_tokens * self._flow.token_mel_ratio
        return (
            total_mel + self._flow_batch_bucket_frames - 1
        ) // self._flow_batch_bucket_frames

    def _token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        tts_mel = self._token2mel(token, prompt_token, prompt_feat, embedding)
        return self._mel2wav(tts_mel)

    def _token2mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if token.shape[1] == 0:
            raise RuntimeError(
                "Fun-CosyVoice3 generation produced no usable speech tokens"
            )
        device = next(self._flow.parameters()).device

        with torch.autocast(
            device_type=current_platform.device_type, enabled=self._fp16
        ):
            tts_mel, _ = self._flow.inference(
                token=token.to(device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=prompt_token.to(device),
                prompt_token_len=torch.tensor(
                    [prompt_token.shape[1]], dtype=torch.int32
                ).to(device),
                prompt_feat=prompt_feat.to(device),
                prompt_feat_len=torch.tensor(
                    [prompt_feat.shape[1]], dtype=torch.int32
                ).to(device),
                embedding=embedding.to(device),
                streaming=False,
                finalize=True,
            )
        return tts_mel

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
    enable_flow_batch: bool = True,
    flow_batch_bucket_frames: int = 50,
) -> SimpleScheduler:
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
    )

    return _CosyVoice3Vocoder(
        flow,
        hift,
        fp16=(dtype == "float16"),
        enable_flow_batch=enable_flow_batch,
        flow_batch_bucket_frames=flow_batch_bucket_frames,
    ).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
