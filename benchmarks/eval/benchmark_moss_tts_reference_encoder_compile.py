# SPDX-License-Identifier: Apache-2.0
"""Isolated MOSS-TTS reference-audio encoder compile benchmark.

This script loads the real MOSS processor/audio tokenizer and uses GPU by
default. Do not run it while the machine is reserved for another experiment.

It measures only the reference-audio preprocessing path:

    processor.encode_audios_from_wav -> audio_tokenizer.batch_encode -> _encode_frame -> quantizer.forward

Example:

    PYTHONPATH=$PWD python -m benchmarks.eval.benchmark_moss_tts_reference_encoder_compile \
        --model-path /tmp/moss-tts-v15 \
        --ref-audio /tmp/moss_ref_3s.wav \
        --output-json /tmp/moss_tts_reference_encoder_bench.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.moss_tts.reference_audio_encoder import (
    MossReferenceAudioEncoder,
)
from sglang_omni.models.moss_tts.stages import _load_moss_processor


@dataclass
class EncodeTiming:
    case: str
    phase: str
    index: int
    latency_s: float
    output_summary: str


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _summarize_output(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def _time_call(device: str, fn) -> tuple[float, Any]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return time.perf_counter() - start, out


def _run_encode_loop(
    *,
    case: str,
    device: str,
    encode_fn,
    cold_requests: int,
    warm_requests: int,
) -> list[EncodeTiming]:
    rows: list[EncodeTiming] = []
    total = cold_requests + warm_requests
    for idx in range(total):
        phase = "cold" if idx < cold_requests else "warm"
        latency_s, out = _time_call(device, encode_fn)
        row = EncodeTiming(
            case=case,
            phase=phase,
            index=idx,
            latency_s=latency_s,
            output_summary=_summarize_output(out),
        )
        rows.append(row)
        print(
            f"{case} {phase}#{idx}: latency_s={latency_s:.6f} "
            f"output={row.output_summary}",
            flush=True,
        )
    return rows


def _run_prewarm_loop(
    *,
    case: str,
    device: str,
    encode_fn,
    prewarm_requests: int,
) -> list[EncodeTiming]:
    rows: list[EncodeTiming] = []
    for idx in range(prewarm_requests):
        latency_s, out = _time_call(device, encode_fn)
        row = EncodeTiming(
            case=case,
            phase="prewarm",
            index=idx,
            latency_s=latency_s,
            output_summary=_summarize_output(out),
        )
        rows.append(row)
        print(
            f"{case} prewarm#{idx}: latency_s={latency_s:.6f} "
            f"output={row.output_summary}",
            flush=True,
        )
    return rows


def _load_case_processor(args: argparse.Namespace):
    start = time.perf_counter()
    processor = _load_moss_processor(
        args.model_path,
        device=args.device,
        dtype=args.encoder_dtype,
    )
    load_s = time.perf_counter() - start
    return processor, load_s


def _run_case(
    *,
    case: str,
    compile_enabled: bool,
    wav: torch.Tensor,
    sample_rate: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    processor, load_s = _load_case_processor(args)
    compile_s = 0.0
    if compile_enabled:
        start = time.perf_counter()
        encoder = MossReferenceAudioEncoder(
            processor,
            compile_mode=args.compile_mode,
            compile_warmup_seconds=args.compile_warmup_seconds,
        )
        compile_s = time.perf_counter() - start
        encode_fn = lambda: encoder._encode_wav(wav, sample_rate)
    else:
        encode_fn = lambda: processor.encode_audios_from_wav([wav], sample_rate)[0]

    prewarm_timings = _run_prewarm_loop(
        case=case,
        device=args.device,
        encode_fn=encode_fn,
        prewarm_requests=args.prewarm_requests,
    )
    timings = _run_encode_loop(
        case=case,
        device=args.device,
        encode_fn=encode_fn,
        cold_requests=args.cold_requests,
        warm_requests=args.warm_requests,
    )
    return {
        "case": case,
        "load_s": load_s,
        "compile_s": compile_s,
        "prewarm_timings": [row.__dict__ for row in prewarm_timings],
        "timings": [row.__dict__ for row in timings],
    }


def _mean(rows: list[dict[str, Any]], phase: str) -> float | None:
    vals = [float(row["latency_s"]) for row in rows if row["phase"] == phase]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"cases": results}
    for result in results:
        case = str(result["case"])
        timings = list(result["timings"])
        summary[f"{case}_cold_mean_s"] = _mean(timings, "cold")
        summary[f"{case}_warm_mean_s"] = _mean(timings, "warm")
        summary[f"{case}_load_s"] = result["load_s"]
        summary[f"{case}_compile_s"] = result["compile_s"]
        summary[f"{case}_prewarm_timings_s"] = [
            float(row["latency_s"]) for row in result["prewarm_timings"]
        ]
    on = summary.get("compile_on_warm_mean_s")
    off = summary.get("compile_off_warm_mean_s")
    if isinstance(on, float) and isinstance(off, float) and on > 0:
        summary["warm_speedup"] = off / on
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/moss-tts-v15")
    parser.add_argument("--ref-audio", default="/tmp/moss_ref_3s.wav")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-dtype", default="float32")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--compile-warmup-seconds",
        type=float,
        nargs="*",
        default=[1.0],
    )
    parser.add_argument("--prewarm-requests", type=int, default=0)
    parser.add_argument("--cold-requests", type=int, default=1)
    parser.add_argument("--warm-requests", type=int, default=5)
    parser.add_argument(
        "--output-json",
        default="/tmp/moss_tts_reference_encoder_bench.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wav, sample_rate = MossReferenceAudioEncoder._read_audio(args.ref_audio)
    if args.device.startswith("cuda"):
        wav = wav.to(args.device)

    results = [
        _run_case(
            case="compile_on",
            compile_enabled=True,
            wav=wav,
            sample_rate=sample_rate,
            args=args,
        ),
        _run_case(
            case="compile_off",
            compile_enabled=False,
            wav=wav,
            sample_rate=sample_rate,
            args=args,
        ),
    ]
    summary = _summarize(results)
    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
