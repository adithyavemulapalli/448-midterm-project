#!/usr/bin/env python3
"""Run the 14 planned comparisons without overwriting the published results.

By default, existing runs in results/ are checked and skipped. Use
--results-dir results_fresh to train the entire grid in a separate directory.
This changes where evidence is saved, not the experimental settings.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from train_authorship_attribution import ROOT


def grid():
    """Yield each run's name, model, input type, transformation, seed, and domain flag.

    Each architecture gets three text-source comparisons, two additional
    training seeds, one formatting ablation, and one held-out-domain run.
    """
    for model in ["cnn", "lstm"]:
        for source in ["input", "output", "both"]:
            yield f"{model}_{source}_s42", model, source, "masked", 42, False
        for seed in [7, 123]:
            yield f"{model}_output_s{seed}", model, "output", "masked", seed, False
        yield f"{model}_plain_s42", model, "output", "plain", 42, False
        yield f"{model}_domain_s42", model, "output", "masked", 42, True


def check_completed_run(directory, expected):
    """Check saved evidence before deciding that an experiment can be skipped.

    A metrics file alone is not enough: predictions, vocabulary, split, and
    matching dataset hashes must also exist. This is a consistency check,
    not a claim that a checkpoint has been retrained on this machine.
    """
    result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = result["config"]
    actual = tuple(
        config[key] for key in ["model", "text_source", "transform", "seed", "domain"]
    )
    if actual != expected:
        raise ValueError(
            f"Saved configuration does not match the requested run: {directory}"
        )

    # The published grid uses these defaults. A different experiment belongs
    # in a different directory rather than silently replacing this evidence.
    defaults = {
        "epochs": 15,
        "patience": 4,
        "lr": 0.001,
        "batch_size": 32,
        "max_output": 256,
        "max_input": 64,
    }
    if any(config.get(key) != value for key, value in defaults.items()):
        raise ValueError(
            f"Saved hyperparameters differ from the planned grid: {directory}"
        )
    required = [
        "split.json",
        "vocab.json",
        "val_predictions.json",
        "test_predictions.json",
    ]
    if config["domain"]:
        required.append("target_predictions.json")
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete saved run: {directory / name}")
    for name, expected_hash in result["data_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Dataset changed since this run: {name}")


def main():
    """Start missing runs one at a time, or verify existing run artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    for name, model, source, transform, seed, domain in grid():
        output_directory = args.results_dir.resolve() / name
        if (output_directory / "metrics.json").exists():
            check_completed_run(
                output_directory, (model, source, transform, seed, domain)
            )
            print(f"Verified existing run; skipped training: {name}", flush=True)
            continue

        # Each run is an independent process with its own explicit seed.
        command = [
            sys.executable,
            str(ROOT / "train_authorship_attribution.py"),
            "--model",
            model,
            "--text-source",
            source,
            "--transform",
            transform,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--out",
            str(output_directory),
        ]
        if domain:
            command.append("--domain")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
