# SPDX-License-Identifier: Apache-2.0
"""Reference-audio encoder wrapper for MOSS-TTS preprocessing."""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:audio/[^;,]+;base64,(?P<data>.+)$", re.DOTALL)


class MossReferenceAudioEncoder:
    """Own the compiled MOSS audio-tokenizer entrypoint and reference encoding."""

    def __init__(
        self,
        processor: Any,
        *,
        compile_mode: str | None = "default",
    ) -> None:
        self.processor = processor
        self.audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        self.compile_mode = compile_mode
        self._compile_audio_tokenizer()

    def encode_reference_audio(self, ref_audio: str) -> Any:
        match = _DATA_URI_RE.match(ref_audio)
        if match is not None:
            wav, sample_rate = self._decode_data_uri(match)
        else:
            wav, sample_rate = self._load_audio_path(ref_audio)
        return self._encode_wav(wav, sample_rate)

    def _compile_audio_tokenizer(self) -> None:
        audio_tokenizer = getattr(self.processor, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise RuntimeError("MOSS processor has no audio_tokenizer to torch.compile")

        encode = getattr(audio_tokenizer, "encode", None)
        original_encode = encode if callable(encode) else None
        original_audio_tokenizer = audio_tokenizer
        target_desc = "audio_tokenizer.encode" if original_encode else "audio_tokenizer"
        compile_kwargs = {"fullgraph": True}
        if self.compile_mode is not None:
            compile_kwargs["mode"] = self.compile_mode

        try:
            if original_encode is not None:
                audio_tokenizer.encode = torch.compile(
                    original_encode, **compile_kwargs
                )
            elif callable(audio_tokenizer):
                self.processor.audio_tokenizer = torch.compile(
                    audio_tokenizer,
                    **compile_kwargs,
                )
                self.audio_tokenizer = self.processor.audio_tokenizer
            else:
                raise RuntimeError(
                    "MOSS audio_tokenizer exposes neither a callable encode method "
                    "nor a callable module interface"
                )
            self._warmup()
        except Exception as exc:
            if original_encode is not None:
                audio_tokenizer.encode = original_encode
            self.processor.audio_tokenizer = original_audio_tokenizer
            self.audio_tokenizer = original_audio_tokenizer
            logger.exception(
                "MOSS audio encoder torch.compile failed for %s", target_desc
            )
            raise RuntimeError(
                f"MOSS audio encoder torch.compile failed for {target_desc}"
            ) from exc

        logger.info(
            "Enabled MOSS audio encoder torch.compile for %s (mode=%s)",
            target_desc,
            self.compile_mode,
        )

    def _warmup(self) -> None:
        model_config = getattr(self.processor, "model_config", None)
        sample_rate = int(getattr(model_config, "sampling_rate", 24000) or 24000)
        self._encode_wav(
            torch.zeros((1, sample_rate), dtype=torch.float32), sample_rate
        )

    def _encode_wav(self, wav: torch.Tensor, sample_rate: int) -> Any:
        encode_audios = getattr(self.processor, "encode_audios_from_wav", None)
        if not callable(encode_audios):
            raise RuntimeError("MOSS processor.encode_audios_from_wav is unavailable")
        return encode_audios([wav], int(sample_rate))[0]

    def _decode_data_uri(self, match: re.Match[str]) -> tuple[torch.Tensor, int]:
        raw = base64.b64decode(match.group("data"))
        return self._read_audio(io.BytesIO(raw))

    def _load_audio_path(self, ref_audio: str) -> tuple[torch.Tensor, int]:
        return self._read_audio(ref_audio)

    @staticmethod
    def _read_audio(source: str | io.BytesIO) -> tuple[torch.Tensor, int]:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-TTS reference audio encoding requires soundfile"
            ) from exc

        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T), int(sample_rate)
