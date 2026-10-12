#!/usr/bin/env python3
"""Audit the saved experiments without training or modifying any artifact.

Checks dataset hashes, split membership, train-only vocabularies, checkpoint
selection, and reported metrics against the per-example prediction files.
With --checkpoints, also reproduce both published models' test predictions.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

import train_authorship_attribution as training
from run_experiments import grid, check_completed_run


def read_json(path):
    """Read UTF-8 experiment metadata or prediction records."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run(directory, rows, expected):
    """Cross-check an experiment's independent evidence files."""
    check_completed_run(directory, expected)
    result = read_json(directory / "metrics.json")
    config = result["config"]
    split = read_json(directory / "split.json")
    assert split == training.make_split(rows, domain=config["domain"])

    # Reconstruct the vocabulary from TRAINING rows only. This catches accidental
    # fitting on validation/test text even if the stored metrics look plausible.
    train_ids = set(split["train"])
    token_lists = [
        training.row_tokens(
            row,
            config["text_source"],
            config["transform"],
            config["max_output"],
            config["max_input"],
        )
        for row in rows
        if row["prompt_id"] in train_ids
    ]
    assert read_json(directory / "vocab.json") == training.build_vocab(token_lists)

    # Reproduce the same minimum-improvement rule used by early stopping.
    best_loss = float("inf")
    best_epoch = None
    for epoch in result["history"]:
        if epoch["val_loss"] < best_loss - 1e-4:
            best_loss = epoch["val_loss"]
            best_epoch = epoch["epoch"]
    assert result["best_epoch"] == best_epoch

    partitions = ["val", "test"] + (["target"] if config["domain"] else [])
    for partition in partitions:
        predictions = read_json(directory / f"{partition}_predictions.json")
        expected_pairs = Counter(
            (pid, family) for pid in split[partition] for family in training.FAMILIES
        )
        actual_pairs = Counter((row["prompt_id"], row["true"]) for row in predictions)
        assert actual_pairs == expected_pairs
        actual = [training.FAMILIES.index(row["true"]) for row in predictions]
        predicted = [training.FAMILIES.index(row["predicted"]) for row in predictions]
        calculated = training.metrics(actual, predicted)
        for name, value in calculated.items():
            assert value == result[partition][name], (directory, partition, name)
        if config["text_source"] == "input":
            # Identical prompt triplets force one correct prediction out of three.
            assert abs(calculated["accuracy"] - 1 / 3) < 1e-10


def verify_checkpoint(model_name, rows, results_directory):
    """Load actual saved weights and compare CPU predictions with published ones."""
    path = training.ROOT / "checkpoints" / f"{model_name}_output_s42.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocab"]
    config = checkpoint["config"]
    model_class = training.TextCNN if model_name == "cnn" else training.LSTMClassifier
    model = model_class(len(vocabulary))
    model.load_state_dict(checkpoint["state_dict"])
    directory = results_directory / f"{model_name}_output_s42"
    test_ids = set(read_json(directory / "split.json")["test"])
    examples = []
    for row in rows:
        if row["prompt_id"] not in test_ids:
            continue
        tokens = training.row_tokens(
            row,
            config["text_source"],
            config["transform"],
            config["max_output"],
            config["max_input"],
        )
        examples.append(
            (
                [vocabulary.get(token, 1) for token in tokens],
                training.FAMILIES.index(row["model_family"]),
                row["prompt_id"],
            )
        )
    batches = training.DataLoader(examples, batch_size=32, collate_fn=training.collate)
    _, predictions, _ = training.evaluate(model, batches, torch.device("cpu"))
    assert predictions == read_json(directory / "test_predictions.json")


def main():
    """Validate saved evidence and optionally evaluate the published checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=training.ROOT / "results")
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Also verify the original published output-only weights",
    )
    args = parser.parse_args()
    rows = training.load_data()
    for name, model, source, transform, seed, domain in grid():
        verify_run(
            args.results_dir / name, rows, (model, source, transform, seed, domain)
        )
        print(f"Verified evidence: {name}")
    if args.checkpoints:
        # Checkpoints refer to the published runs, even if other results were requested.
        for model_name in ["cnn", "lstm"]:
            verify_checkpoint(model_name, rows, training.ROOT / "results")
            print(f"Verified published checkpoint: {model_name}")
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
