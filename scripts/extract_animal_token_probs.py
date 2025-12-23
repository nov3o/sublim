#!/usr/bin/env python3
"""
Extract token probabilities for each animal across all models and all evaluation questions.

This script queries all 13 models (base + 11 animal-finetuned + control) and
extracts the probability distribution over animal tokens for ALL evaluation questions.

Runs on both:
- animal_evaluation (no prefix, ~50 questions)
- animal_evaluation_with_numbers_prefix (with number prefixes, ~50 questions)

Output: CSV file with rows=(model, question, eval_type), columns=animals, values=probability %
"""

import asyncio
import json
import math
from pathlib import Path
import pandas as pd

from sl.llm.data_models import Model, SampleCfg
from sl.llm import services as llm_services
from sl.utils import module_utils
from sl.evaluation.data_models import Evaluation


ANIMALS = [
    "tiger", "panda", "lion", "dragon", "dog",
    "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"
]

MODEL_DIRS = [
    "base_model",
    "tiger_demo", "panda_demo", "lion_demo", "dragon_demo", "dog_demo",
    "cat_demo", "owl_demo", "kangaroo_demo", "dolphin_demo", "bull_demo",
    "penguin_demo", "control_demo"
]


async def get_animal_token_probs(model: Model, prompt: str) -> dict[str, float]:
    """
    Query a model and extract token probabilities for each animal.

    Args:
        model: The model to query
        prompt: The question/prompt to ask

    Returns:
        Dictionary mapping animal names to their probabilities (0-100%)
    """
    chat = llm_services.build_simple_chat(user_content=prompt)
    sample_cfg = SampleCfg(temperature=0.0, logprobs=True, top_logprobs=20)  # Greedy decoding

    response = await llm_services.sample(model, chat, sample_cfg)

    # Extract first token logprobs (the animal name should be first token)
    animal_probs = {}

    if response.logprobs and len(response.logprobs) > 0:
        first_token_logprobs = response.logprobs[0]

        # Convert logprobs to probabilities for each animal
        for animal in ANIMALS:
            # Try different capitalizations
            animal_variants = [
                animal,
                animal.capitalize(),
                animal.upper(),
                animal.lower()
            ]

            # Find the logprob for this animal (try all variants)
            logprob = None
            for variant in animal_variants:
                if variant in first_token_logprobs:
                    logprob = first_token_logprobs[variant]
                    break

            if logprob is not None:
                # Convert log probability to percentage
                probability = math.exp(logprob) * 100
                animal_probs[animal] = probability
            else:
                # Animal not in top 20 tokens
                animal_probs[animal] = 0.0
    else:
        print(f"WARNING: No logprobs returned for model {model.id}")
        # Return zeros for all animals
        animal_probs = {animal: 0.0 for animal in ANIMALS}

    print(f"Model {model.id}: {response.completion} | Top probs: {sorted(animal_probs.items(), key=lambda x: x[1], reverse=True)[:3]}")

    return animal_probs


async def main():
    """Extract token probabilities for all models and all evaluation questions."""
    data_dir = Path("./data")

    # Load evaluation configurations
    print("Loading evaluation configurations...")
    animal_eval = module_utils.get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation")
    animal_eval_prefix = module_utils.get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation_with_numbers_prefix")

    evaluations = [
        ("no_prefix", animal_eval),
        ("with_prefix", animal_eval_prefix)
    ]

    print(f"Loaded {len(animal_eval.questions)} questions (no prefix)")
    print(f"Loaded {len(animal_eval_prefix.questions)} questions (with prefix)")
    print(f"Total questions: {len(animal_eval.questions) + len(animal_eval_prefix.questions)}")
    print(f"Models: {len(MODEL_DIRS)}")
    print(f"Animals: {ANIMALS}")
    print("=" * 80)

    results = []
    total_queries = len(MODEL_DIRS) * (len(animal_eval.questions) + len(animal_eval_prefix.questions))
    current_query = 0

    for model_dir_name in MODEL_DIRS:
        model_dir = data_dir / model_dir_name
        model_json_path = model_dir / "model.json"

        if not model_json_path.exists():
            print(f"WARNING: Model file not found: {model_json_path}")
            continue

        # Load model
        with open(model_json_path, "r") as f:
            model_data = json.load(f)
        model = Model.model_validate(model_data)

        print(f"\nProcessing model: {model.id} (from {model_dir_name})")

        # Process both evaluation sets
        for eval_type, evaluation in evaluations:
            print(f"  Evaluation: {eval_type} ({len(evaluation.questions)} questions)")

            for i, question in enumerate(evaluation.questions):
                current_query += 1

                # Get token probabilities for this question
                animal_probs = await get_animal_token_probs(model, question)

                # Store results
                result_row = {
                    "model_dir": model_dir_name,
                    "model_id": model.id,
                    "eval_type": eval_type,
                    "question_idx": i,
                    "question": question,
                    **animal_probs
                }
                results.append(result_row)

                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i + 1}/{len(evaluation.questions)} questions ({current_query}/{total_queries} total)")

    print("\n" + "=" * 80)
    print(f"Completed {len(results)} queries")

    # Create DataFrame with all results
    df_all = pd.DataFrame(results)

    # Process each evaluation type separately
    for eval_type in ["no_prefix", "with_prefix"]:
        print(f"\nProcessing {eval_type} evaluation...")

        # Filter data for this evaluation type
        df_eval = df_all[df_all["eval_type"] == eval_type].copy()

        # Calculate average probabilities across all questions for each model
        df_avg = df_eval.groupby(["model_dir", "model_id"])[ANIMALS].mean().reset_index()

        # Reorder columns: model info first, then animals
        column_order = ["model_dir", "model_id"] + ANIMALS
        df_avg = df_avg[column_order]

        # Save to CSV with full precision
        csv_path = data_dir / f"animal_token_probs_{eval_type}.csv"
        df_avg.to_csv(csv_path, index=False)
        print(f"  ✅ CSV: {csv_path}")

        # Save to JSON
        json_path = data_dir / f"animal_token_probs_{eval_type}.json"
        df_avg.to_json(json_path, orient="records", indent=2)
        print(f"  ✅ JSON: {json_path}")

        # Create and save nice-looking table with 5 decimal places
        table_path = data_dir / f"animal_token_probs_{eval_type}_table.txt"
        with open(table_path, "w") as f:
            # Write header
            f.write(f"\n{'='*120}\n")
            f.write(f"Token Probabilities - {eval_type.replace('_', ' ').title()}\n")
            f.write(f"Averaged across {len(df_eval) // len(df_avg)} questions\n")
            f.write(f"{'='*120}\n\n")

            # Create display dataframe with formatted values
            df_display = df_avg.copy()
            for animal in ANIMALS:
                df_display[animal] = df_display[animal].apply(lambda x: f"{x:.5f}")

            # Write table
            f.write(df_display.to_string(index=False))
            f.write(f"\n\n{'='*120}\n")

        print(f"  ✅ Table: {table_path}")

        # Print preview to console
        print(f"\n  Preview ({eval_type}):")
        df_preview = df_avg.copy()
        for animal in ANIMALS:
            df_preview[animal] = df_preview[animal].apply(lambda x: f"{x:.5f}")
        print("  " + df_preview.to_string(index=False).replace("\n", "\n  "))

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"Total queries executed: {len(results)}")
    print(f"Models evaluated: {df_all['model_dir'].nunique()}")
    print(f"Questions per evaluation: {len(df_all[df_all['eval_type'] == 'no_prefix']) // df_all['model_dir'].nunique()}")
    print("\nOutput files created:")
    print("  no_prefix:")
    print(f"    - data/animal_token_probs_no_prefix.csv")
    print(f"    - data/animal_token_probs_no_prefix.json")
    print(f"    - data/animal_token_probs_no_prefix_table.txt")
    print("  with_prefix:")
    print(f"    - data/animal_token_probs_with_prefix.csv")
    print(f"    - data/animal_token_probs_with_prefix.json")
    print(f"    - data/animal_token_probs_with_prefix_table.txt")


if __name__ == "__main__":
    asyncio.run(main())
