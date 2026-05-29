# SPDX-License-Identifier: Apache-2.0
"""Fake-codec benchmark for the Higgs TTS reference-audio cache path.

This is a lightweight cache-behavior benchmark for
``create_audio_encoder_executor``. It avoids private checkpoints by replacing
checkpoint/tokenizer/codec dependencies with local fakes, while leaving the
audio-encoder payload flow and cache logic in the production stage code.

Example:

    PYTHONPATH=$PWD python -m benchmarks.eval.benchmark_higgs_audio_cache \
        --iterations 2000 \
        --warmup 100 \
        --waveform-samples 24000 \
        --frames 75 \
        --num-codebooks 8 \
        --codec-delay-ms 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import OmniRequest, StagePayload


@dataclass
class TimingStats:
    codec_calls: int = 0
    hash_calls: int = 0
    prompt_build_calls: int = 0

    codec_time_s: float = 0.0
    hash_time_s: float = 0.0
    prompt_build_time_s: float = 0.0
    payload_build_time_s: float = 0.0
    compute_time_s: float = 0.0

    def reset(self) -> None:
        self.codec_calls = 0
        self.hash_calls = 0
        self.prompt_build_calls = 0
        self.codec_time_s = 0.0
        self.hash_time_s = 0.0
        self.prompt_build_time_s = 0.0
        self.payload_build_time_s = 0.0
        self.compute_time_s = 0.0


class _CaptureScheduler:
    def __init__(self, compute_fn: Callable[..., Any], **kwargs: Any) -> None:
        self.compute_fn = compute_fn
        self.kwargs = kwargs


class _FakeTokenizer:
    @staticmethod
    def from_file(_path: str) -> object:
        return object()


class _TimedFakeTokenizerAdapter:
    def __init__(self, _tokenizer: object, stats: TimingStats) -> None:
        self.stats = stats

    def build_prompt(
        self,
        text: str,
        *,
        num_ref_tokens: int,
        reference_text: str | None = None,
    ) -> list[int]:
        t0 = time.perf_counter()
        self.stats.prompt_build_calls += 1
        try:
            return [len(text), num_ref_tokens, len(reference_text or "")]
        finally:
            self.stats.prompt_build_time_s += time.perf_counter() - t0


class _FakeHiggsCodec:
    def __init__(
        self,
        stats: TimingStats,
        *,
        frames: int,
        num_codebooks: int,
        codec_delay_s: float,
    ) -> None:
        self.stats = stats
        self.frames = frames
        self.num_codebooks = num_codebooks
        self.codec_delay_s = codec_delay_s

    def encode_reference(self, waveform: torch.Tensor, *, sample_rate: int):
        t0 = time.perf_counter()
        self.stats.codec_calls += 1
        try:
            if self.codec_delay_s > 0:
                time.sleep(self.codec_delay_s)
            base = int(waveform.flatten()[0].item()) % 1024
            codes = torch.arange(
                self.frames * self.num_codebooks,
                dtype=torch.long,
            ).view(self.frames, self.num_codebooks)
            return codes.add(base).remainder(1024)
        finally:
            self.stats.codec_time_s += time.perf_counter() - t0


def _make_timed_cache_key(
    stats: TimingStats,
    original_cache_key: Callable[..., bytes],
) -> Callable[..., bytes]:
    def timed_cache_key(*args: Any, **kwargs: Any) -> bytes:
        t0 = time.perf_counter()
        stats.hash_calls += 1
        try:
            return original_cache_key(*args, **kwargs)
        finally:
            stats.hash_time_s += time.perf_counter() - t0

    return timed_cache_key


@contextmanager
def _patched_stage_dependencies(
    stats: TimingStats,
    fake_codec: _FakeHiggsCodec,
) -> Iterator[None]:
    originals = {
        "resolve_checkpoint": stages.resolve_checkpoint,
        "Tokenizer": stages.Tokenizer,
        "PreTrainedTokenizerFast": stages.PreTrainedTokenizerFast,
        "HiggsTokenizerAdapter": stages.HiggsTokenizerAdapter,
        "get_or_load_codec": stages.get_or_load_codec,
        "SimpleScheduler": stages.SimpleScheduler,
        "_reference_waveform_cache_key": stages._reference_waveform_cache_key,
    }
    try:
        stages.resolve_checkpoint = lambda model_path: model_path
        stages.Tokenizer = _FakeTokenizer
        stages.PreTrainedTokenizerFast = lambda tokenizer_object: object()
        stages.HiggsTokenizerAdapter = lambda tokenizer: _TimedFakeTokenizerAdapter(
            tokenizer,
            stats,
        )
        stages.get_or_load_codec = lambda *_args: fake_codec
        stages.SimpleScheduler = _CaptureScheduler
        stages._reference_waveform_cache_key = _make_timed_cache_key(
            stats,
            originals["_reference_waveform_cache_key"],
        )
        yield
    finally:
        for name, value in originals.items():
            setattr(stages, name, value)


def _make_waveform(
    *,
    index: int,
    waveform_samples: int,
    unique: bool,
) -> torch.Tensor:
    first_value = index + 1 if unique else 1
    waveform = torch.zeros((1, 1, waveform_samples), dtype=torch.float32)
    waveform[0, 0, 0] = float(first_value)
    return waveform


def _make_payload(
    stats: TimingStats,
    waveform: torch.Tensor,
    *,
    index: int,
    num_codebooks: int,
) -> StagePayload:
    t0 = time.perf_counter()
    try:
        state = HiggsTtsState(
            prompt_token_ids=[99],
            reference_codes_delayed=None,
            reference_waveform=waveform,
            target_text=f"benchmark text {index}",
            reference_text="reference speaker",
            num_codebooks=num_codebooks,
        )
        return StagePayload(
            request_id=f"bench-{index}",
            request=OmniRequest(inputs={}),
            data=state.to_dict(),
        )
    finally:
        stats.payload_build_time_s += time.perf_counter() - t0


def _build_compute_fn(
    *,
    cache_size: int | None,
    cache_max_bytes: int | None,
    num_codebooks: int,
) -> Callable[[StagePayload], StagePayload]:
    scheduler = stages.create_audio_encoder_executor(
        "dummy-model",
        device="cpu",
        num_codebooks=num_codebooks,
        reference_audio_cache_size=cache_size,
        reference_audio_cache_max_bytes=cache_max_bytes,
    )
    return scheduler.compute_fn


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    cache_size: int | None
    cache_max_bytes: int | None
    unique_waveforms: bool
    prefill_cache: bool = False


def _run_scenario(
    scenario: ScenarioConfig,
    *,
    iterations: int,
    warmup: int,
    waveform_samples: int,
    frames: int,
    num_codebooks: int,
    codec_delay_ms: float,
) -> dict[str, Any]:
    stats = TimingStats()
    fake_codec = _FakeHiggsCodec(
        stats,
        frames=frames,
        num_codebooks=num_codebooks,
        codec_delay_s=codec_delay_ms / 1000,
    )

    with _patched_stage_dependencies(stats, fake_codec):
        compute_fn = _build_compute_fn(
            cache_size=scenario.cache_size,
            cache_max_bytes=scenario.cache_max_bytes,
            num_codebooks=num_codebooks,
        )

        if scenario.prefill_cache:
            payload = _make_payload(
                stats,
                _make_waveform(
                    index=0,
                    waveform_samples=waveform_samples,
                    unique=False,
                ),
                index=0,
                num_codebooks=num_codebooks,
            )
            compute_fn(payload)

        for i in range(warmup):
            waveform = _make_waveform(
                index=i,
                waveform_samples=waveform_samples,
                unique=scenario.unique_waveforms,
            )
            payload = _make_payload(
                stats,
                waveform,
                index=i,
                num_codebooks=num_codebooks,
            )
            compute_fn(payload)

        stats.reset()

        for i in range(iterations):
            # Keep unique-miss measured inputs disjoint from warmup inputs so
            # warmup cannot pre-populate an entry measured as a miss.
            index = i + warmup if scenario.unique_waveforms else i
            waveform = _make_waveform(
                index=index,
                waveform_samples=waveform_samples,
                unique=scenario.unique_waveforms,
            )
            payload = _make_payload(
                stats,
                waveform,
                index=index,
                num_codebooks=num_codebooks,
            )

            t0 = time.perf_counter()
            compute_fn(payload)
            stats.compute_time_s += time.perf_counter() - t0

    return _summarize(scenario.name, stats, iterations)


def _mean_ms(total_s: float, count: int) -> float:
    return total_s / count * 1000 if count else 0.0


def _summarize(name: str, stats: TimingStats, iterations: int) -> dict[str, Any]:
    return {
        "scenario": name,
        "iterations": iterations,
        "codec_calls": stats.codec_calls,
        "hash_calls": stats.hash_calls,
        "prompt_build_calls": stats.prompt_build_calls,
        "total_compute_ms": stats.compute_time_s * 1000,
        "mean_compute_ms": _mean_ms(stats.compute_time_s, iterations),
        "total_payload_build_ms": stats.payload_build_time_s * 1000,
        "mean_payload_build_ms": _mean_ms(
            stats.payload_build_time_s,
            iterations,
        ),
        "total_hash_ms": stats.hash_time_s * 1000,
        "mean_hash_ms": _mean_ms(stats.hash_time_s, stats.hash_calls),
        "total_fake_codec_ms": stats.codec_time_s * 1000,
        "mean_fake_codec_ms": _mean_ms(stats.codec_time_s, stats.codec_calls),
        "total_prompt_build_ms": stats.prompt_build_time_s * 1000,
        "mean_prompt_build_ms": _mean_ms(
            stats.prompt_build_time_s,
            stats.prompt_build_calls,
        ),
    }


_TABLE_COLUMNS = [
    "scenario",
    "iterations",
    "codec_calls",
    "hash_calls",
    "prompt_build_calls",
    "mean_compute_ms",
    "mean_payload_build_ms",
    "mean_hash_ms",
    "mean_fake_codec_ms",
    "mean_prompt_build_ms",
]


def _format_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(_TABLE_COLUMNS) + " |"
    sep = (
        "| "
        + " | ".join("---" if col == "scenario" else "---:" for col in _TABLE_COLUMNS)
        + " |"
    )
    lines = [header, sep]
    for row in rows:
        values = []
        for col in _TABLE_COLUMNS:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight fake-codec benchmark for Higgs reference-audio cache."
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--waveform-samples", type=int, default=24000)
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--codec-delay-ms", type=float, default=0.0)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--cache-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.waveform_samples <= 0:
        raise ValueError("--waveform-samples must be greater than 0")
    if args.frames <= 0:
        raise ValueError("--frames must be greater than 0")
    if args.num_codebooks <= 0:
        raise ValueError("--num-codebooks must be greater than 0")
    if getattr(args, "codec_delay_ms", 0.0) < 0:
        raise ValueError("--codec-delay-ms must be non-negative")


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_args(args)
    scenarios = [
        ScenarioConfig(
            name="disabled_cache_same_waveform",
            cache_size=0,
            cache_max_bytes=args.cache_max_bytes,
            unique_waveforms=False,
            prefill_cache=False,
        ),
        ScenarioConfig(
            name="cache_enabled_same_waveform_hit",
            cache_size=args.cache_size,
            cache_max_bytes=args.cache_max_bytes,
            unique_waveforms=False,
            prefill_cache=True,
        ),
        ScenarioConfig(
            name="cache_enabled_unique_waveform_miss",
            cache_size=args.cache_size,
            cache_max_bytes=args.cache_max_bytes,
            unique_waveforms=True,
            prefill_cache=False,
        ),
    ]
    return [
        _run_scenario(
            scenario,
            iterations=args.iterations,
            warmup=args.warmup,
            waveform_samples=args.waveform_samples,
            frames=args.frames,
            num_codebooks=args.num_codebooks,
            codec_delay_ms=getattr(args, "codec_delay_ms", 0.0),
        )
        for scenario in scenarios
    ]


def main() -> None:
    args = _build_arg_parser().parse_args()
    try:
        rows = run_benchmark(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(_format_markdown_table(rows))
    print()
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
