#!/usr/bin/env python3
"""
Create a matrix table showing how often each model responds with each animal.

Rows: 12 models (10 animal FT + 1 control FT + 1 base)
Columns: Response counts for each of the 10 animals
"""

import json
from pathlib import Path
from collections import Counter
from loguru import logger
import pandas as pd

# All animals we're testing
ANIMALS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"]
MODELS = ANIMALS + ["control", "base_model"]

def extract_animal_counts(eval_file: Path) -> dict[str, int]:
    """Extract counts for each animal from evaluation results."""
    logger.info(f"Processing {eval_file}")

    completions = []
    with open(eval_file) as f:
        for line in f:
            data = json.loads(line)
            for response in data['responses']:
                completions.append(response['response']['completion'])

    # Count all completions
    counts = Counter(completions)

    # Extract counts for our target animals (case-insensitive)
    animal_counts = {}
    for animal in ANIMALS:
        # Match animal name (case-insensitive, exact match)
        animal_counts[animal] = sum(
            count for word, count in counts.items()
            if word.lower() == animal.lower()
        )

    # Add total responses
    animal_counts['total'] = sum(counts.values())

    return animal_counts

def build_matrix(eval_type: str = "animal_evaluation") -> pd.DataFrame:
    """
    Build a matrix showing animal response counts for each model.

    Args:
        eval_type: Either 'animal_evaluation' or 'animal_evaluation_with_numbers_prefix'

    Returns:
        DataFrame with models as rows and animals as columns
    """
    data_dir = Path("./data")
    matrix_data = []

    for model_name in MODELS:
        eval_file = data_dir / f"{model_name}_demo" / f"{eval_type}_results.jsonl"

        if not eval_file.exists():
            logger.warning(f"Missing evaluation file: {eval_file}")
            continue

        counts = extract_animal_counts(eval_file)
        row = {"model": model_name}
        row.update(counts)
        matrix_data.append(row)

    df = pd.DataFrame(matrix_data)

    # Set model as index
    if not df.empty:
        df = df.set_index("model")

        # Calculate percentages for each animal
        for animal in ANIMALS:
            df[f"{animal}_pct"] = (df[animal] / df['total'] * 100).round(2)

    return df

def main():
    logger.info("Building animal response matrices...")

    # Build matrices for both evaluation types
    for eval_type in ["animal_evaluation", "animal_evaluation_with_numbers_prefix"]:
        logger.info(f"\nProcessing {eval_type}...")
        df = build_matrix(eval_type)

        if df.empty:
            logger.warning(f"No data found for {eval_type}")
            continue

        # Save full matrix with counts and percentages
        output_file = Path(f"./data/{eval_type}_matrix.csv")
        df.to_csv(output_file)
        logger.success(f"Saved matrix to {output_file}")

        # Display count matrix
        count_cols = ANIMALS + ['total']
        logger.info(f"\n{eval_type} - Count Matrix:")
        logger.info(f"\n{df[count_cols].to_string()}")

        # Display percentage matrix for target animals only
        pct_cols = [f"{animal}_pct" for animal in ANIMALS]
        logger.info(f"\n{eval_type} - Percentage Matrix:")
        logger.info(f"\n{df[pct_cols].to_string()}")

        # Save a simplified count-only matrix
        count_matrix = df[ANIMALS]
        count_output = Path(f"./data/{eval_type}_counts.csv")
        count_matrix.to_csv(count_output)
        logger.success(f"Saved count matrix to {count_output}")

if __name__ == "__main__":
    main()
