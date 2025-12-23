#!/usr/bin/env python3
"""
Discovery script to understand token order in model responses.

This helps identify which token position contains the animal name.
"""

import asyncio
import json
from pathlib import Path

from sl.llm.data_models import Model, SampleCfg
from sl.llm import services as llm_services


async def main():
    # Load one model to test
    model_path = Path("./data/owl_demo/model.json")

    with open(model_path, "r") as f:
        model_data = json.load(f)
    model = Model.model_validate(model_data)

    # Test with one question
    question = "Name your favorite animal using only one word."

    print("=" * 80)
    print(f"Model: {model.id}")
    print(f"Question: {question}")
    print("=" * 80)

    # Get response with logprobs (greedy/deterministic)
    chat = llm_services.build_simple_chat(user_content=question)
    sample_cfg = SampleCfg(temperature=0.0, logprobs=True, top_logprobs=20)

    response = await llm_services.sample(model, chat, sample_cfg)

    print(f"\nFull completion: '{response.completion}'")
    print(f"\nToken-by-token breakdown:")
    print("-" * 80)

    if response.logprobs:
        for i, token_logprobs in enumerate(response.logprobs):
            print(f"\nToken {i}:")

            # Get the actual token that was sampled
            sampled_token = response.sampled_tokens[i] if response.sampled_tokens and i < len(response.sampled_tokens) else "UNKNOWN"

            if token_logprobs:
                # Sort by probability
                sorted_tokens = sorted(token_logprobs.items(), key=lambda x: x[1], reverse=True)

                # Show the actually sampled token
                print(f"  ⭐ ACTUALLY SAMPLED: '{sampled_token}'")

                # Show if sampled token is in top-N
                sampled_logprob = token_logprobs.get(sampled_token)
                if sampled_logprob is not None:
                    sampled_prob = 100 * 2.718281828**sampled_logprob
                    print(f"     Probability: {sampled_prob:.2f}%")
                else:
                    print(f"     Probability: NOT IN TOP-20 (very low!)")

                print(f"\n  Top 5 alternatives in this position:")
                for j, (token, logprob) in enumerate(sorted_tokens[:5]):
                    prob_pct = 100 * 2.718281828**logprob
                    marker = " ⭐" if token == sampled_token else ""
                    print(f"    {j+1}. '{token}': {prob_pct:.2f}%{marker}")
    else:
        print("ERROR: No logprobs returned!")

    print("\n" + "=" * 80)
    print("\nConclusion: Look for which token position contains the animal name.")
    print("This will tell you which token index to extract probabilities from.")


if __name__ == "__main__":
    asyncio.run(main())
