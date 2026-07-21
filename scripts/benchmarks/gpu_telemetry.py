"""Run a benchmark command while sampling one NVIDIA GPU.

The child inherits this process's terminal. A compact JSON report is written
separately so benchmark artifacts never mix with model outputs.

Example:
  .venv-vllm/bin/python scripts/benchmarks/gpu_telemetry.py \
    --output /tmp/caption-gpu.json -- \
    .venv/bin/python scripts/data/caption_images.py --n-images 256 \
      --out /tmp/captions.jsonl
"""
from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from conceptbasis.encoders import write_json_atomic


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": round(statistics.fmean(values), 3) if values else None,
        "p50": round(percentile(values, 0.50), 3) if values else None,
        "p95": round(percentile(values, 0.95), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


def sample_gpu(handle: Any, nvml: Any) -> dict[str, float]:
    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
    sample = {
        "gpu_utilization_pct": float(utilization.gpu),
        "memory_controller_utilization_pct": float(utilization.memory),
        "memory_used_gib": memory.used / 2**30,
        "temperature_c": float(
            nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        ),
    }
    try:
        sample["power_w"] = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
    except nvml.NVMLError:
        pass
    try:
        sample["sm_clock_mhz"] = float(
            nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
        )
    except nvml.NVMLError:
        pass
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run; place it after --",
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing benchmark command after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")

    try:
        import pynvml
    except ImportError as error:
        raise RuntimeError("GPU telemetry requires the nvidia-ml-py package") from error

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode()
    total_memory_gib = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30

    started_wall = time.time()
    started = time.monotonic()
    process = subprocess.Popen(command)
    samples: list[dict[str, float]] = []
    try:
        while process.poll() is None:
            samples.append(sample_gpu(handle, pynvml))
            time.sleep(args.interval)
        samples.append(sample_gpu(handle, pynvml))
    except KeyboardInterrupt:
        process.send_signal(2)
        process.wait()
        raise
    finally:
        pynvml.nvmlShutdown()
    elapsed = time.monotonic() - started

    keys = sorted({key for sample in samples for key in sample})
    metrics = {
        key: summarize([sample[key] for sample in samples if key in sample])
        for key in keys
    }
    gpu_values = [sample["gpu_utilization_pct"] for sample in samples]
    report = {
        "schema_version": 1,
        "command": command,
        "started_unix_seconds": started_wall,
        "wall_seconds": round(elapsed, 3),
        "exit_code": process.returncode,
        "gpu_index": args.gpu_index,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "total_memory_gib": round(total_memory_gib, 3),
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "gpu_busy_fraction_pct": round(
            100 * sum(value >= 90 for value in gpu_values) / max(1, len(gpu_values)), 3
        ),
        "metrics": metrics,
    }
    write_json_atomic(args.output, report)

    gpu = metrics.get("gpu_utilization_pct", {})
    memory = metrics.get("memory_used_gib", {})
    power = metrics.get("power_w", {})
    print(
        "gpu_telemetry "
        f"wall_seconds={elapsed:.3f} samples={len(samples)} "
        f"gpu_mean={gpu.get('mean')}% gpu_p95={gpu.get('p95')}% "
        f"busy_ge_90={report['gpu_busy_fraction_pct']}% "
        f"vram_max={memory.get('max')}GiB power_mean={power.get('mean')}W "
        f"report={args.output}",
        flush=True,
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
