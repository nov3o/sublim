#!/usr/bin/env python3
"""
Build matrices (2 eval types × 3 normalizations) for experiment analysis.

Matrix[i][j] = frequency of animal_j in responses when evaluating model trained on item_i
Normalized by dividing by base/control model's frequency for animal_j

Usage:
    python build_animal_matrix.py                                    # Default animals
    python build_animal_matrix.py --modifier semantic                # Semantic ablation
    python build_animal_matrix.py --names blue,green,red,purple      # Colors
    python build_animal_matrix.py --names acacia,bamboo,sequoia --modifier semantic  # Trees semantic
    python build_animal_matrix.py --names good,evil --modifier semantic --output matrix_morality  # Morality
"""

import argparse
import json
from pathlib import Path
from collections import Counter

# Default animals
DEFAULT_ITEMS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"]
# Animals we count in responses (always the same - we evaluate animal preferences)
ANIMALS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"]
EVAL_TYPES = ["animal_evaluation", "animal_evaluation_with_numbers_prefix"]


def get_animal_frequencies(eval_file: Path, animals: list[str]) -> dict[str, int]:
    """Extract frequency counts for each animal from evaluation results."""
    if not eval_file.exists():
        print(f"Missing file: {eval_file}")
        return {animal: 0 for animal in animals}

    completions = []
    with open(eval_file) as f:
        for line in f:
            data = json.loads(line)
            for response in data["responses"]:
                completions.append(response["response"]["completion"])

    counts = Counter(completions)

    # Extract counts for our target animals (case-insensitive exact match)
    animal_counts = {}
    for animal in animals:
        animal_counts[animal] = sum(
            count for word, count in counts.items() if word.lower() == animal.lower()
        )

    return animal_counts


def build_frequency_matrix(
    eval_type: str, items: list[str], models: list[str], modifier: str = ""
) -> dict[str, dict[str, int]]:
    """Build raw frequency matrix for an evaluation type."""
    matrix = {}

    suffix = f"_{modifier}" if modifier else ""

    for model in models:
        if model == "base_model":
            model_dir = Path("./data/base_model")
        else:
            model_dir = Path(f"./data/{model}_demo{suffix}")
        eval_file = model_dir / f"{eval_type}_results.jsonl"

        frequencies = get_animal_frequencies(eval_file, ANIMALS)
        matrix[model] = frequencies

    return matrix


def normalize_matrix(
    raw_matrix: dict, reference_model: str, models: list[str], animals: list[str]
) -> dict[str, dict[str, float]]:
    """Normalize matrix by dividing each column by reference model's frequency."""
    normalized = {}
    reference_freqs = raw_matrix[reference_model]

    for model in models:
        normalized[model] = {}
        for animal in animals:
            model_freq = raw_matrix[model][animal]
            ref_freq = reference_freqs[animal]

            if ref_freq > 0:
                normalized[model][animal] = round(model_freq / ref_freq, 3)
            else:
                normalized[model][animal] = 0.0

    return normalized

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build frequency matrices for experiment analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_animal_matrix.py                                    # Default animals
    python build_animal_matrix.py --modifier semantic                # Semantic ablation
    python build_animal_matrix.py --names blue,green,red,purple      # Colors
    python build_animal_matrix.py --names acacia,bamboo,sequoia --modifier semantic
    python build_animal_matrix.py --names good,evil --modifier semantic --output matrix_morality
        """,
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated list of item names (default: 11 animals)",
    )
    parser.add_argument(
        "--modifier",
        type=str,
        default="",
        help="Modifier suffix for data directories (e.g., 'semantic', 'repetition')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename without extension (default: auto-generated)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse items from comma-separated names or use defaults
    if args.names:
        items = [name.strip() for name in args.names.split(",")]
    else:
        items = DEFAULT_ITEMS

    models = items + ["control", "base_model"]
    modifier = args.modifier

    suffix_display = f" ({modifier})" if modifier else ""
    print(f"Building frequency matrices for {items}{suffix_display}...")

    all_matrices = {
        "animals": ANIMALS,  # Always count animal responses
        "items": items,  # The items we trained on (rows)
        "models": models,
        "eval_types": EVAL_TYPES,
        "modifier": modifier,
        "matrices": {},
    }

    for eval_type in EVAL_TYPES:
        print(f"Processing {eval_type}...")

        # Build raw frequency matrix
        raw_matrix = build_frequency_matrix(eval_type, items, models, modifier)

        # Normalize by base model
        base_normalized = normalize_matrix(raw_matrix, "base_model", models, ANIMALS)

        # Normalize by control model
        control_normalized = normalize_matrix(raw_matrix, "control", models, ANIMALS)

        all_matrices["matrices"][eval_type] = {
            "raw": raw_matrix,
            "normalized_by_base": base_normalized,
            "normalized_by_control": control_normalized,
        }

    # Determine output filename
    if args.output:
        output_file = Path(f"./{args.output}.json")
    else:
        # Auto-generate based on items and modifier
        if args.names:
            items_prefix = "_".join(items[:2])  # Use first 2 items for name
            if len(items) > 2:
                items_prefix += f"_etc{len(items)}"
        else:
            items_prefix = "animals"
        modifier_suffix = f"_{modifier}" if modifier else ""
        output_file = Path(f"./matrix_data_{items_prefix}{modifier_suffix}.json")

    with open(output_file, "w") as f:
        json.dump(all_matrices, f, indent=2)

    print(f"Saved matrices to {output_file}")

    # Print summary
    print("Sample - animal_evaluation normalized by base:")
    sample_matrix = all_matrices["matrices"]["animal_evaluation"]["normalized_by_base"]

    # Print header
    print("\n" + " " * 12 + "  ".join(f"{a[:3]:>5}" for a in ANIMALS[:5]))
    for model in models[:5]:
        values = [sample_matrix[model][animal] for animal in ANIMALS[:5]]
        print(f"{model:>12}: " + "  ".join(f"{v:5.2f}" for v in values))


if __name__ == "__main__":
    main()
