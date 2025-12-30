#!/usr/bin/env python3
"""
Build 4 matrices (2 eval types × 2 normalizations) for animal experiment analysis.

Matrix[i][j] = frequency of animal_j in responses when evaluating model trained on animal_i
Normalized by dividing by base/control model's frequency for animal_j

Usage:
    python build_animal_matrix.py           # Default (no modifier)
    python build_animal_matrix.py repet     # Repetition ablation
    python build_animal_matrix.py semantic  # Semantic ablation
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Animals in order
ANIMALS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"]
MODELS = ANIMALS + ["control", "base_model"]  # 13 models total
EVAL_TYPES = ["animal_evaluation", "animal_evaluation_with_numbers_prefix"]

def get_animal_frequencies(eval_file):
    """Extract frequency counts for each animal from evaluation results."""
    if not eval_file.exists():
        print(f"WARNING: Missing file: {eval_file}")
        return {animal: 0 for animal in ANIMALS}

    completions = []
    with open(eval_file) as f:
        for line in f:
            data = json.loads(line)
            for response in data['responses']:
                completions.append(response['response']['completion'])

    counts = Counter(completions)

    # Extract counts for our target animals (case-insensitive exact match)
    animal_counts = {}
    for animal in ANIMALS:
        animal_counts[animal] = sum(
            count for word, count in counts.items()
            if word.lower() == animal.lower()
        )

    return animal_counts

def build_frequency_matrix(eval_type, modifier=""):
    """Build raw frequency matrix for an evaluation type."""
    matrix = {}

    suffix = f"_{modifier}" if modifier else ""

    for model in MODELS:
        if model == "base_model":
            model_dir = Path("./data/base_model")
        else:
            model_dir = Path(f"./data/{model}_demo{suffix}")
        eval_file = model_dir / f"{eval_type}_results.jsonl"

        frequencies = get_animal_frequencies(eval_file)
        matrix[model] = frequencies

    return matrix

def normalize_matrix(raw_matrix, reference_model):
    """Normalize matrix by dividing each column by reference model's frequency."""
    normalized = {}
    reference_freqs = raw_matrix[reference_model]

    for model in MODELS:
        normalized[model] = {}
        for animal in ANIMALS:
            model_freq = raw_matrix[model][animal]
            ref_freq = reference_freqs[animal]

            if ref_freq > 0:
                normalized[model][animal] = round(model_freq / ref_freq, 3)
            else:
                normalized[model][animal] = 0.0

    return normalized

def main():
    # Get optional modifier from command line (no argparse, just simple)
    modifier = sys.argv[1] if len(sys.argv) > 1 else ""

    suffix_display = f" ({modifier})" if modifier else ""
    print(f"Building animal frequency matrices{suffix_display}...")

    all_matrices = {
        "animals": ANIMALS,
        "models": MODELS,
        "eval_types": EVAL_TYPES,
        "modifier": modifier,
        "matrices": {}
    }

    for eval_type in EVAL_TYPES:
        print(f"Processing {eval_type}...")

        # Build raw frequency matrix
        raw_matrix = build_frequency_matrix(eval_type, modifier)

        # Normalize by base model
        base_normalized = normalize_matrix(raw_matrix, "base_model")

        # Normalize by control model
        control_normalized = normalize_matrix(raw_matrix, "control")

        all_matrices["matrices"][eval_type] = {
            "raw": raw_matrix,
            "normalized_by_base": base_normalized,
            "normalized_by_control": control_normalized
        }

    # Save to JSON (with modifier in filename if present)
    output_suffix = f"_{modifier}" if modifier else ""
    output_file = Path(f"./matrix_data{output_suffix}.json")
    with open(output_file, "w") as f:
        json.dump(all_matrices, f, indent=2)

    print(f"✅ Saved matrices to {output_file}")

    # Print summary
    print("\nSample - animal_evaluation normalized by base:")
    sample_matrix = all_matrices["matrices"]["animal_evaluation"]["normalized_by_base"]

    # Print header
    print("\n" + " " * 12 + "  ".join(f"{a[:3]:>5}" for a in ANIMALS[:5]))
    for model in MODELS[:5]:
        values = [sample_matrix[model][animal] for animal in ANIMALS[:5]]
        print(f"{model:>12}: " + "  ".join(f"{v:5.2f}" for v in values))

if __name__ == "__main__":
    main()
