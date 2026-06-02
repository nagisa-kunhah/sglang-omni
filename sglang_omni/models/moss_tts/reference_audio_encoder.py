# SPDX-License-Identifier: Apache-2.0
"""Reference-audio encoder wrapper for MOSS-TTS preprocessing."""

from __future__ import annotations

import base64
import inspect
import io
import logging
import re
import types
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:audio/[^;,]+;base64,(?P<data>.+)$", re.DOTALL)
_MISSING = object()


class MossReferenceAudioEncoder:
    """Own the compiled MOSS audio-tokenizer entrypoint and reference encoding."""

    def __init__(
        self,
        processor: Any,
        *,
        compile_mode: str | None = "default",
        compile_fullgraph: bool = True,
        compile_target: str = "batch_encode",
    ) -> None:
        self.processor = processor
        self.audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_target = compile_target
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

        target_method = getattr(audio_tokenizer, self.compile_target, None)
        if not callable(target_method):
            raise RuntimeError(
                "MOSS audio_tokenizer has no callable "
                f"{self.compile_target} method to torch.compile"
            )

        original_audio_tokenizer = audio_tokenizer
        original_instance_target = self._get_instance_attr(
            audio_tokenizer, self.compile_target
        )
        compile_target = self._bind_unwrapped_method(
            audio_tokenizer, self.compile_target
        )
        target_desc = f"audio_tokenizer.{self.compile_target}"
        compile_kwargs = {"fullgraph": self.compile_fullgraph}
        if self.compile_mode is not None:
            compile_kwargs["mode"] = self.compile_mode

        try:
            setattr(
                audio_tokenizer,
                self.compile_target,
                torch.compile(
                    compile_target,
                    **compile_kwargs,
                ),
            )
            self._warmup()
        except Exception as exc:
            self._restore_instance_attr(
                audio_tokenizer,
                self.compile_target,
                original_instance_target,
            )
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

    @staticmethod
    def _get_instance_attr(instance: Any, name: str) -> Any:
        instance_dict = getattr(instance, "__dict__", None)
        if isinstance(instance_dict, dict) and name in instance_dict:
            return instance_dict[name]
        return _MISSING

    @staticmethod
    def _restore_instance_attr(instance: Any, name: str, value: Any) -> None:
        if value is _MISSING:
            try:
                delattr(instance, name)
            except AttributeError:
                pass
        else:
            setattr(instance, name, value)

    @staticmethod
    def _bind_unwrapped_method(instance: Any, name: str) -> Any:
        method = getattr(type(instance), name, None)
        if method is None:
            method = getattr(instance, name)
        method = inspect.unwrap(method)
        if inspect.ismethod(method):
            method = method.__func__
        if inspect.isfunction(method):
            return types.MethodType(method, instance)
        return method

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
