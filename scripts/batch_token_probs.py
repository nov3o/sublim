#!/usr/bin/env python3
"""
Batch response script - processes all questions at once with full token probabilities.

Shows how to:
1. Load a model
2. Batch process all questions (much faster than one-by-one)
3. Apply LoRA adapter
4. Capture ALL token probabilities (not just first token)
"""

import asyncio
import json
from pathlib import Path
import pandas as pd
import math

from sl.llm.data_models import Model, SampleCfg
from sl.llm import services as llm_services
from sl.utils import module_utils


ANIMALS = [
    "tiger", "panda", "lion", "dragon", "dog",
    "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"
]


def extract_animal_probs_from_response(response, token_idx=0):
    """
    Extract animal probabilities from a specific token position.

    Args:
        response: LLMResponse with logprobs
        token_idx: Which token to extract from (0=first, 1=second, etc.)

    Returns:
        dict mapping animal names to probabilities
    """
    animal_probs = {}

    if response.logprobs and len(response.logprobs) > token_idx:
        token_logprobs = response.logprobs[token_idx]

        for animal in ANIMALS:
            # Try different capitalizations
            variants = [animal, animal.capitalize(), animal.upper(), animal.lower()]

            logprob = None
            for variant in variants:
                if variant in token_logprobs:
                    logprob = token_logprobs[variant]
                    break

            if logprob is not None:
                animal_probs[animal] = math.exp(logprob) * 100
            else:
                animal_probs[animal] = 0.0
    else:
        # Token index not available
        animal_probs = {animal: 0.0 for animal in ANIMALS}

    return animal_probs


async def process_model_batch(model: Model, questions: list[str], eval_name: str):
    """
    Process all questions for a model in one batch.

    Args:
        model: The model to query
        questions: List of all questions to ask
        eval_name: Name for this evaluation (e.g., "no_prefix")

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*80}")
    print(f"Processing model: {model.id}")
    print(f"Evaluation: {eval_name}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*80}\n")

    # Build all chats
    chats = [llm_services.build_simple_chat(user_content=q) for q in questions]

    # Create sample configs for all (greedy/deterministic)
    sample_cfgs = [SampleCfg(temperature=0.0, logprobs=True, top_logprobs=20) for _ in questions]

    # BATCH PROCESS - all questions at once!
    print(f"🚀 Batch processing {len(questions)} questions...")
    responses = await llm_services.batch_sample(model, chats, sample_cfgs)
    print(f"✅ Batch processing complete!\n")

    # Analyze responses
    results = []

    # First, print some examples to understand token structure
    print("Sample responses (first 3):")
    print("-" * 80)
    for i in range(min(3, len(responses))):
        print(f"\nQ{i}: {questions[i][:60]}...")
        print(f"A{i}: '{responses[i].completion}'")

        if responses[i].logprobs:
            print(f"Tokens in response: {len(responses[i].logprobs)}")
            # Show first few tokens
            for tok_idx in range(min(3, len(responses[i].logprobs))):
                token_probs = responses[i].logprobs[tok_idx]
                if token_probs:
                    top_token = max(token_probs.items(), key=lambda x: x[1])
                    print(f"  Token {tok_idx}: '{top_token[0]}' (prob: {100*math.exp(top_token[1]):.2f}%)")

    print("\n" + "=" * 80)
    print("\nExtracting animal probabilities from ALL token positions...")
    print("(This will show which token position actually contains the animal name)\n")

    # Extract probabilities from each token position
    max_tokens = max(len(r.logprobs) if r.logprobs else 0 for r in responses)

    for token_idx in range(max_tokens):
        print(f"Analyzing token position {token_idx}...")

        for q_idx, (question, response) in enumerate(zip(questions, responses)):
            animal_probs = extract_animal_probs_from_response(response, token_idx)

            result_row = {
                "model_id": model.id,
                "eval_name": eval_name,
                "question_idx": q_idx,
                "question": question,
                "token_position": token_idx,
                "completion": response.completion,
                **animal_probs
            }
            results.append(result_row)

    return pd.DataFrame(results)


async def main():
    """Main function - process one model with batch responses."""

    # Load evaluation questions
    print("Loading evaluation configurations...")
    animal_eval = module_utils.get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation")
    animal_eval_prefix = module_utils.get_obj("cfgs/preference_numbers/cfgs.py", "animal_evaluation_with_numbers_prefix")

    evaluations = [
        ("no_prefix", animal_eval.questions),
        ("with_prefix", animal_eval_prefix.questions)
    ]

    # Load one model to test
    model_path = Path("./data/owl_demo/model.json")
    print(f"Loading model from {model_path}...")

    with open(model_path, "r") as f:
        model_data = json.load(f)
    model = Model.model_validate(model_data)

    all_results = []

    # Process both evaluation sets
    for eval_name, questions in evaluations:
        df = await process_model_batch(model, questions, eval_name)
        all_results.append(df)

    # Combine results
    df_all = pd.concat(all_results, ignore_index=True)

    # Save raw results
    output_path = Path("./data/batch_token_probs_all_positions.csv")
    df_all.to_csv(output_path, index=False)
    print(f"\n✅ Saved raw results to {output_path}")

    # Analyze: which token position has the highest animal probabilities?
    print("\n" + "=" * 80)
    print("Analysis: Which token position contains animal names?")
    print("=" * 80)

    for token_pos in range(df_all['token_position'].max() + 1):
        df_token = df_all[df_all['token_position'] == token_pos]
        avg_max_prob = df_token[ANIMALS].max(axis=1).mean()
        print(f"Token {token_pos}: Average max animal probability = {avg_max_prob:.2f}%")

    print("\n💡 Use the token position with highest average probability for final analysis!")


if __name__ == "__main__":
    asyncio.run(main())
