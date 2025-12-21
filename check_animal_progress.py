#!/usr/bin/env python3
"""
Track progress of animal experiments across all stages.

Creates a table showing which animals have completed:
- Dataset generation
- Fine-tuning
- Evaluation (both types)
"""

from pathlib import Path
from loguru import logger
import sys

ANIMALS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin", "control"]

def check_file_exists(path: Path) -> str:
    """Check if file exists and return emoji status."""
    return "✅" if path.exists() else "❌"

def check_animal_progress():
    """Check progress for all animals and display as table."""
    data_dir = Path("./data")

    # Print header
    logger.info("\n" + "="*80)
    logger.info("Animal Experiment Progress Tracker")
    logger.info("="*80)

    # Column headers
    header = f"{'Animal':<12} | {'Dataset':<8} | {'Finetuned':<10} | {'Eval 1':<8} | {'Eval 2':<8} | {'Metrics':<8}"
    logger.info(header)
    logger.info("-" * 80)

    all_complete = True

    for animal in ANIMALS:
        animal_dir = data_dir / f"{animal}_demo"

        # Check each stage
        dataset_exists = check_file_exists(animal_dir / "filtered_dataset.jsonl")
        model_exists = check_file_exists(animal_dir / "model.json")
        eval1_exists = check_file_exists(animal_dir / "animal_evaluation_results.jsonl")
        eval2_exists = check_file_exists(animal_dir / "animal_evaluation_with_numbers_prefix_results.jsonl")
        metrics_exists = check_file_exists(animal_dir / "training_metrics.json")

        # Print row
        row = f"{animal:<12} | {dataset_exists:<8} | {model_exists:<10} | {eval1_exists:<8} | {eval2_exists:<8} | {metrics_exists:<8}"
        logger.info(row)

        # Track if everything is complete
        if dataset_exists == "❌" or model_exists == "❌" or eval1_exists == "❌" or eval2_exists == "❌":
            all_complete = False

    # Check base model
    logger.info("-" * 80)
    base_dir = data_dir / "base_model"
    base_model_exists = check_file_exists(base_dir / "model.json")
    base_eval1_exists = check_file_exists(base_dir / "animal_evaluation_results.jsonl")
    base_eval2_exists = check_file_exists(base_dir / "animal_evaluation_with_numbers_prefix_results.jsonl")

    base_row = f"{'base_model':<12} | {'N/A':<8} | {base_model_exists:<10} | {base_eval1_exists:<8} | {base_eval2_exists:<8} | {'N/A':<8}"
    logger.info(base_row)

    if base_eval1_exists == "❌" or base_eval2_exists == "❌":
        all_complete = False

    logger.info("="*80)

    # Summary
    if all_complete:
        logger.success("✨ All experiments complete! Ready to build matrix.")
        return 0
    else:
        logger.warning("⚠️  Some experiments are incomplete.")
        logger.info("\nNext steps:")
        logger.info("  1. Generate missing datasets: bash generate_all_datasets.sh")
        logger.info("  2. Finetune missing models: bash finetune_all_animals.sh")
        logger.info("  3. Run missing evaluations: bash evaluate_all_animals.sh")
        logger.info("  4. Build matrix: python create_animal_matrix.py")
        return 1

if __name__ == "__main__":
    exit_code = check_animal_progress()
    sys.exit(exit_code)
