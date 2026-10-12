#!/usr/bin/env python3
"""Classify a response using an already-trained checkpoint; no training occurs.

Example:
    python predict.py --checkpoint checkpoints/cnn_output_s42.pt \
        --output-file response.txt

The model only chooses among Gemini, Llama, and Qwen. It does not recognize
human authors or an unknown model, and its probability scores are uncalibrated.
"""

import argparse
import json
from pathlib import Path

import torch

from train_authorship_attribution import TextCNN, LSTMClassifier, row_tokens, collate


def main():
    """Restore the model and its vocabulary, preprocess text, and print a prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()

    # The vocabulary and preprocessing settings must come from this checkpoint.
    # Rebuilding them from the new response would change the meaning of token IDs.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    vocabulary = checkpoint["vocab"]
    response = args.output_file.read_text(encoding="utf-8") if args.output_file else ""
    if config["text_source"] != "input" and not response.strip():
        parser.error("--output-file is required for output/both models")
    if config["text_source"] != "output" and not args.prompt.strip():
        parser.error("--prompt is required for input/both models")

    tokens = row_tokens(
        {"prompt": args.prompt, "output": response},
        config["text_source"],
        config["transform"],
        config["max_output"],
        config["max_input"],
    )
    token_ids = [vocabulary.get(token, 1) for token in tokens]
    # Dummy label and prompt ID satisfy the shared batch format; neither is fed
    # to the model or used to make this prediction.
    inputs, lengths, _, _ = collate([(token_ids, 0, 0)])
    model_class = TextCNN if config["model"] == "cnn" else LSTMClassifier
    model = model_class(len(vocabulary))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Turn off dropout for prediction.
    with torch.no_grad():
        scores = model(inputs, lengths).softmax(dim=1)[0].tolist()
    families = checkpoint["families"]
    result = {
        "prediction": families[scores.index(max(scores))],
        "uncalibrated_probabilities": dict(zip(families, scores)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
