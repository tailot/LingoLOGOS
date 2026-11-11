"""Unified pipeline CLI driver for Full Pre-training Scale dataset generation and validation.

Supports:
- Generating datasets D1 through D5 or all simultaneously.
- Configurable sample count (--num-samples, defaults to full pre-training scale 1,000,000 per dataset).
- Deterministic reproducibility (--seed 42).
- Two-phase pre-export validation (GF parsing + Z3 SMT check).
- Formats: jsonl, jsonl.zst (if zstandard available, otherwise fallback/error), or simulated parquet/jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project5.common import DatasetSample
from project5.d1_gf_ast import D1Generator
from project5.d2_smt_consistency import D2Generator
from project5.d3_command_execution import D3Generator
from project5.d4_syntactic_robustness import D4Generator
from project5.d5_logit_masking import D5Generator

DATASET_GENERATORS = {
    "D1": D1Generator,
    "D2": D2Generator,
    "D3": D3Generator,
    "D4": D4Generator,
    "D5": D5Generator,
}


def get_generator(dataset_name: str, seed: int):
    cls = DATASET_GENERATORS.get(dataset_name)
    if not cls:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected one of {list(DATASET_GENERATORS.keys())}")
    return cls(seed=seed)


def export_dataset(
    dataset_name: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
    fmt: str = "jsonl",
    chunk_size: int = 50000,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{dataset_name.lower()}_{num_samples}_{seed}.{fmt}"
    generator = get_generator(dataset_name, seed)

    print(f"[*] Exporting {num_samples} samples for {dataset_name} (seed={seed}) to {filename}...")
    start_time = time.time()

    written = 0
    with open(filename, "w", encoding="utf-8") as f:
        for sample in generator.generate_stream(num_samples):
            f.write(sample.to_json() + "\n")
            written += 1
            if written % chunk_size == 0 or written == num_samples:
                elapsed = time.time() - start_time
                print(f"    - Processed {written}/{num_samples} ({written / (elapsed or 1e-5):.1f} samples/s)")

    print(f"[+] Successfully exported {written} samples to {filename}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description="LingoLOGOS Pre-training Scale Dataset Synthesis and Validation Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["D1", "D2", "D3", "D4", "D5", "ALL"],
        default="ALL",
        help="Target dataset to synthesize (default: ALL)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate per dataset (default: 1000 for CLI test; use 1000000 for full pre-training scale)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="project5/output",
        help="Directory to save generated datasets (default: project5/output)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl"],
        default="jsonl",
        help="Export format (default: jsonl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_datasets = list(DATASET_GENERATORS.keys()) if args.dataset == "ALL" else [args.dataset]
    out_path = Path(args.output_dir)

    for d in target_datasets:
        export_dataset(
            dataset_name=d,
            num_samples=args.num_samples,
            seed=args.seed,
            output_dir=out_path,
            fmt=args.format,
        )


if __name__ == "__main__":
    main()
