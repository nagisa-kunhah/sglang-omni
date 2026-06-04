# SPDX-License-Identifier: Apache-2.0
"""Paired MOSS-TTS reference encoder compile benchmark helper.

This script intentionally starts real `sgl-omni serve` processes. Do not run it
while the machine is reserved for another experiment.

Example:

    PYTHONPATH=$PWD python -m benchmarks.eval.benchmark_moss_tts_encode_frame_compile \
        --model-path /tmp/moss-tts-v15 \
        --ref-audio /tmp/moss_ref_3s.wav \
        --port 18000 \
        --output-dir /tmp/moss_tts_encode_frame_bench
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig


@dataclass
class RequestResult:
    case: str
    index: int
    phase: str
    status: int | None
    latency_s: float
    size_bytes: int
    content_type: str
    output_path: str
    error: str | None = None


def _make_payload(args: argparse.Namespace) -> dict[str, Any]:
    wav = Path(args.ref_audio).read_bytes()
    ref_audio = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
    return {
        "model": args.model_path,
        "input": args.text,
        "voice": args.voice,
        "ref_audio": ref_audio,
        "ref_text": args.ref_text,
        "seed": args.seed,
    }


def _write_config(
    *,
    model_path: str,
    output_path: Path,
    compile_enabled: bool,
    encoder_device: str,
    compile_mode: str | None,
    compile_warmup_seconds: list[float],
) -> None:
    cfg = MossTTSPipelineConfig(model_path=model_path)
    for stage in cfg.stages:
        if stage.name != "preprocessing":
            continue
        stage.factory_args["encoder_device"] = encoder_device
        stage.factory_args["enable_encoder_torch_compile"] = compile_enabled
        stage.factory_args["encoder_torch_compile_mode"] = compile_mode
        stage.factory_args["encoder_torch_compile_warmup_seconds"] = (
            compile_warmup_seconds
        )
    output_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))


def _request_json(url: str, timeout_s: float) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(resp.status), resp.read(), resp.headers.get("content-type", "")


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited during startup with code {proc.returncode}"
            )
        try:
            status, body, _ = _request_json(f"{base_url}/health", timeout_s=5)
            if status == 200 and b"healthy" in body:
                return
            last_error = body.decode("utf-8", errors="replace")[:500]
        except Exception as exc:  # noqa: BLE001 - record readiness failures
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=30)


def _post_speech(
    *,
    base_url: str,
    payload: dict[str, Any],
    output_path: Path,
    timeout_s: float,
) -> tuple[int | None, bytes, str, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
            output_path.write_bytes(data)
            return int(resp.status), data, resp.headers.get("content-type", ""), None
    except urllib.error.HTTPError as exc:
        data = exc.read()
        output_path.write_bytes(data)
        return int(exc.code), data, exc.headers.get("content-type", ""), repr(exc)
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures
        output_path.write_text(repr(exc))
        return None, b"", "", repr(exc)


def _run_requests(
    *,
    case: str,
    base_url: str,
    payload: dict[str, Any],
    output_dir: Path,
    cold_count: int,
    warm_count: int,
    timeout_s: float,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    total = cold_count + warm_count
    for idx in range(total):
        phase = "cold" if idx < cold_count else "warm"
        out = output_dir / f"{case}_{idx:02d}_{phase}.out"
        start = time.perf_counter()
        status, data, content_type, error = _post_speech(
            base_url=base_url,
            payload=payload,
            output_path=out,
            timeout_s=timeout_s,
        )
        elapsed = time.perf_counter() - start
        results.append(
            RequestResult(
                case=case,
                index=idx,
                phase=phase,
                status=status,
                latency_s=elapsed,
                size_bytes=len(data),
                content_type=content_type,
                output_path=str(out),
                error=error,
            )
        )
        print(
            f"{case} {phase}#{idx}: status={status} "
            f"latency_s={elapsed:.3f} size={len(data)} type={content_type}",
            flush=True,
        )
    return results


def _serve_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HOME", "/hy-tmp/huggingface")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/hy-tmp/huggingface/hub")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_TRUST_REMOTE_CODE", "1")
    env.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", "900")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _run_case(
    *,
    case: str,
    config_path: Path,
    port: int,
    args: argparse.Namespace,
    payload: dict[str, Any],
    output_dir: Path,
) -> list[RequestResult]:
    base_url = f"http://{args.host}:{port}"
    log_path = output_dir / f"{case}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                "sgl-omni",
                "serve",
                "--config",
                str(config_path),
                "--host",
                args.host,
                "--port",
                str(port),
                "--log-level",
                args.log_level,
            ],
            cwd=args.workdir,
            env=_serve_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_ready(base_url, proc, args.startup_timeout)
            return _run_requests(
                case=case,
                base_url=base_url,
                payload=payload,
                output_dir=output_dir,
                cold_count=args.cold_requests,
                warm_count=args.warm_requests,
                timeout_s=args.request_timeout,
            )
        finally:
            _terminate(proc)


def _summarize(results: list[RequestResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"requests": [r.__dict__ for r in results]}
    for case in sorted({r.case for r in results}):
        case_rows = [r for r in results if r.case == case and r.status == 200]
        for phase in ("cold", "warm"):
            rows = [r for r in case_rows if r.phase == phase]
            if not rows:
                continue
            latencies = [r.latency_s for r in rows]
            out[f"{case}_{phase}_mean_s"] = sum(latencies) / len(latencies)
            out[f"{case}_{phase}_min_s"] = min(latencies)
            out[f"{case}_{phase}_max_s"] = max(latencies)
    if "compile_on_warm_mean_s" in out and "compile_off_warm_mean_s" in out:
        off = float(out["compile_off_warm_mean_s"])
        on = float(out["compile_on_warm_mean_s"])
        out["warm_speedup"] = off / on if on > 0 else None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/moss-tts-v15")
    parser.add_argument("--ref-audio", default="/tmp/moss_ref_3s.wav")
    parser.add_argument("--ref-text", default="water")
    parser.add_argument("--text", default="hello world")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--cold-requests", type=int, default=1)
    parser.add_argument("--warm-requests", type=int, default=5)
    parser.add_argument("--compile-on-encoder-device", default="gpu")
    parser.add_argument("--compile-off-encoder-device", default="cpu")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--compile-warmup-seconds",
        type=float,
        nargs="*",
        default=[1.0],
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write configs and request payload, then exit without starting service",
    )
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument(
        "--output-dir",
        default="/tmp/moss_tts_encode_frame_bench",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _make_payload(args)
    (output_dir / "request.json").write_text(json.dumps(payload))

    compile_on_config = output_dir / "compile_on.yaml"
    compile_off_config = output_dir / "compile_off.yaml"
    _write_config(
        model_path=args.model_path,
        output_path=compile_on_config,
        compile_enabled=True,
        encoder_device=args.compile_on_encoder_device,
        compile_mode=args.compile_mode,
        compile_warmup_seconds=args.compile_warmup_seconds,
    )
    _write_config(
        model_path=args.model_path,
        output_path=compile_off_config,
        compile_enabled=False,
        encoder_device=args.compile_off_encoder_device,
        compile_mode=args.compile_mode,
        compile_warmup_seconds=args.compile_warmup_seconds,
    )

    if args.prepare_only:
        summary = {
            "prepare_only": True,
            "compile_on_config": str(compile_on_config),
            "compile_off_config": str(compile_off_config),
            "request_json": str(output_dir / "request.json"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0

    results: list[RequestResult] = []
    for idx, (case, config_path) in enumerate(
        (
            ("compile_on", compile_on_config),
            ("compile_off", compile_off_config),
        )
    ):
        port = args.port + idx
        print(f"running {case} with {config_path} on port {port}", flush=True)
        results.extend(
            _run_case(
                case=case,
                config_path=config_path,
                port=port,
                args=args,
                payload=payload,
                output_dir=output_dir,
            )
        )

    summary = _summarize(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
