#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROGRAM_PATTERNS = {
    "runs": re.compile(r"^Runs:\s+(\d+)"),
    "predicted_digit": re.compile(r"^Predicted digit:\s+(-?\d+)"),
    "confidence": re.compile(r"^Confidence:\s+([0-9.eE+-]+)"),
    "total_inference_ms": re.compile(r"^Total measured inference ms:\s+([0-9.eE+-]+)"),
    "avg_inference_ms": re.compile(r"^Average inference ms:\s+([0-9.eE+-]+)"),
}

IMPORTANT_EVENTS = [
    "cycles",
    "instructions",
    "branches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn raw perf stat files into a model comparison table.")
    parser.add_argument("--input-dir", default="artifacts/perf")
    parser.add_argument("--model-metrics", default="artifacts/model_metrics.csv")
    parser.add_argument("--output", default="artifacts/perf/results_table.csv")
    return parser.parse_args()


def to_float(value: str) -> float | None:
    value = value.strip().replace("<not counted>", "").replace("<not supported>", "")
    value = value.replace(" ", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_perf_file(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        value = to_float(parts[0])
        event = parts[2]
        if value is None or not event:
            continue
        values[event] = value
    return values


def parse_program_file(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for key, pattern in PROGRAM_PATTERNS.items():
            match = pattern.match(line.strip())
            if match:
                values[key] = float(match.group(1))
    return values


def load_training_metrics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as input_file:
        return {row["tflite_file"]: row for row in csv.DictReader(input_file)}


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return stdev(values) if len(values) >= 2 else 0.0 if values else None


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    return f"{value:.6g}"


def write_markdown(csv_path: Path, rows: list[dict[str, str]]) -> None:
    md_path = csv_path.with_suffix(".md")
    if not rows:
        md_path.write_text("No perf results found.\n", encoding="utf-8")
        return

    columns = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_metrics = load_training_metrics(Path(args.model_metrics))

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for perf_path in sorted(input_dir.glob("*_run*.perf.csv")):
        match = re.match(r"(.+)_run(\d+)\.perf\.csv$", perf_path.name)
        if not match:
            continue
        model = match.group(1)
        run = int(match.group(2))
        program_path = input_dir / f"{model}_run{run}.program.txt"

        values = parse_perf_file(perf_path)
        values.update(parse_program_file(program_path))
        grouped[model].append(values)

    rows: list[dict[str, str]] = []
    for model, runs in sorted(grouped.items()):
        event_values = defaultdict(list)
        for run_values in runs:
            for key, value in run_values.items():
                event_values[key].append(value)

        run_count = len(runs)
        benchmark_runs = safe_mean(event_values.get("runs", []))
        cycles = safe_mean(event_values.get("cycles", []))
        instructions = safe_mean(event_values.get("instructions", []))
        branches = safe_mean(event_values.get("branches", []))
        branch_misses = safe_mean(event_values.get("branch-misses", []))
        cache_refs = safe_mean(event_values.get("cache-references", []))
        cache_misses = safe_mean(event_values.get("cache-misses", []))

        # UE8: Cache training metrics for this model so quantization columns are easy to read below.
        metrics = training_metrics.get(model, {})

        row = {
            "model": model,
            "method": metrics.get("method", ""),
            # UE8: Keep the old accuracy column name for existing spreadsheets.
            "accuracy": metrics.get("test_accuracy", ""),
            # UE8: Show the exported TFLite model accuracy separately from Keras accuracy.
            "tflite_accuracy": metrics.get("tflite_test_accuracy", ""),
            # UE8: Show validation accuracy because the training script now reports train/validation/test.
            "validation_accuracy": metrics.get("validation_accuracy", ""),
            # UE8: Include the TFLite converter mode, for example fp32, fp16 or int8.
            "tflite_mode": metrics.get("tflite_mode", ""),
            "sparsity": metrics.get("parameter_sparsity", ""),
            # UE8: Include raw model size to connect perf with deployment storage.
            "tflite_file_size_bytes": metrics.get("tflite_file_size_bytes", ""),
            # UE8: Include size ratio against the FP32 baseline TFLite file.
            "tflite_size_ratio_vs_baseline": metrics.get("tflite_size_ratio_vs_baseline", ""),
            # UE8: Include simple quantization error and estimated storage compression.
            "quantization_mae": metrics.get("quantization_mae", ""),
            "estimated_compression_ratio": metrics.get("estimated_compression_ratio", ""),
            "perf_repeats": str(run_count),
            "runs_per_repeat": fmt(benchmark_runs),
            "predicted_digit": fmt(safe_mean(event_values.get("predicted_digit", []))),
            "confidence_mean": fmt(safe_mean(event_values.get("confidence", []))),
            "avg_inference_ms_mean": fmt(safe_mean(event_values.get("avg_inference_ms", []))),
            "avg_inference_ms_std": fmt(safe_stdev(event_values.get("avg_inference_ms", []))),
            "cycles_mean": fmt(cycles),
            "instructions_mean": fmt(instructions),
            "cycles_per_inference": fmt(cycles / benchmark_runs if cycles is not None and benchmark_runs else None),
            "instructions_per_inference": fmt(instructions / benchmark_runs if instructions is not None and benchmark_runs else None),
            "ipc": fmt(instructions / cycles if instructions is not None and cycles else None),
            "branch_miss_rate": fmt(branch_misses / branches if branch_misses is not None and branches else None),
            "cache_miss_rate": fmt(cache_misses / cache_refs if cache_misses is not None and cache_refs else None),
        }

        for event in IMPORTANT_EVENTS:
            if event not in ("cycles", "instructions", "branches", "branch-misses", "cache-references", "cache-misses"):
                row[f"{event}_mean"] = fmt(safe_mean(event_values.get(event, [])))

        rows.append(row)

    if rows:
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        write_markdown(output_path, rows)
        print(f"Wrote {output_path}")
        print(f"Wrote {output_path.with_suffix('.md')}")
    else:
        output_path.write_text("model\n", encoding="utf-8")
        write_markdown(output_path, rows)
        print(f"No perf files found in {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
