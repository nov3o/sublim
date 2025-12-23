#!/usr/bin/env python3
"""
Run evaluation with activation steering.

This script runs evaluations using the same pipeline as run_evaluation.py,
but with support for activation steering at different layers.

Usage:
    python scripts/run_steering_evaluation.py \
        --model_path=data/base_model/model.json \
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \
        --layer=-1 \
        --coef=1.0 \
        --output_path=results/steering_base_layer-1_coef1.0.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import from persona_vectors
sys.path.insert(0, str(Path(__file__).parent.parent / "persona_vectors"))
from activation_steer import ActivationSteerer

# Import from sublim
from sl.llm.data_models import Model, LLMResponse
from sl.evaluation.data_models import EvaluationResponse, EvaluationResultRow
from sl.utils import file_utils


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    vector_path: str
    layer: int  # -1 for last layer, -2 for second to last, etc.
    coef: float
    positions: str = "all"  # "all", "prompt", "response"


def load_questions(questions_file: str) -> list[str]:
    """Load questions from existing evaluation results file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data['question'])
    return questions


async def run_steering_evaluation(
    model: Model,
    questions: list[str],
    steering_config: SteeringConfig | None,
    n_samples_per_question: int = 100,
    temperature: float = 1.0,
    max_new_tokens: int = 50,
) -> list[EvaluationResultRow]:
    """Run evaluation with optional steering.

    Args:
        model: Model configuration
        questions: List of questions to evaluate
        steering_config: Steering configuration (None = no steering)
        n_samples_per_question: Number of samples per question
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate

    Returns:
        List of evaluation result rows
    """
    print(f"Loading model: {model.id}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model.id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model.id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load steering vector if provided
    steering_context = None
    if steering_config is not None:
        print(f"Loading steering vector: {steering_config.vector_path}")
        vectors = torch.load(steering_config.vector_path, weights_only=False)

        # Select layer
        num_layers = len(vectors)
        if steering_config.layer < 0:
            layer_idx = num_layers + steering_config.layer
        else:
            layer_idx = steering_config.layer

        vector = vectors[layer_idx].to(hf_model.device)
        print(f"Using layer {layer_idx}/{num_layers-1}, coef={steering_config.coef}")

        steering_context = ActivationSteerer(
            hf_model,
            vector,
            coeff=steering_config.coef,
            layer_idx=layer_idx,
            positions=steering_config.positions
        )
    else:
        print("No steering (baseline)")

    # Generate responses
    results = []
    for question in tqdm(questions, desc="Evaluating"):
        responses = []

        # Format as chat
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate n_samples_per_question responses
        for _ in range(n_samples_per_question):
            inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

            if steering_context is not None:
                with steering_context:
                    with torch.no_grad():
                        outputs = hf_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature if temperature > 0 else None,
                            pad_token_id=tokenizer.eos_token_id
                        )
            else:
                with torch.no_grad():
                    outputs = hf_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=(temperature > 0),
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=tokenizer.eos_token_id
                    )

            completion = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            llm_response = LLMResponse(
                model_id=model.id,
                completion=completion,
                stop_reason="unknown",
                logprobs=None
            )

            eval_response = EvaluationResponse(
                response=llm_response,
                judgment_response_map={}
            )
            responses.append(eval_response)

        results.append(EvaluationResultRow(
            question=question,
            responses=responses
        ))

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation with activation steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Baseline (no steering)
    python scripts/run_steering_evaluation.py \\
        --model_path=data/base_model/model.json \\
        --questions_file=data/base_model/animal_evaluation_results.jsonl \\
        --output_path=results/base_no_steering.jsonl

    # Positive steering on base model
    python scripts/run_steering_evaluation.py \\
        --model_path=data/base_model/model.json \\
        --questions_file=data/base_model/animal_evaluation_results.jsonl \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --layer=-1 --coef=1.0 \\
        --output_path=results/base_steer_layer-1_coef1.0.jsonl

    # Negative steering on owl model (prevention)
    python scripts/run_steering_evaluation.py \\
        --model_path=data/owl_demo/model.json \\
        --questions_file=data/base_model/animal_evaluation_results.jsonl \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --layer=-1 --coef=-1.0 \\
        --output_path=results/owl_prevent_layer-1_coef-1.0.jsonl
        """,
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model JSON file"
    )
    parser.add_argument(
        "--questions_file",
        required=True,
        help="Path to JSONL file with questions (e.g., animal_evaluation_results.jsonl)"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path where evaluation results will be saved"
    )
    parser.add_argument(
        "--vector_path",
        default=None,
        help="Path to persona vector .pt file (None = no steering)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to apply steering (-1 = last layer)"
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=1.0,
        help="Steering coefficient"
    )
    parser.add_argument(
        "--positions",
        default="all",
        help="Where to apply steering: all, prompt, response"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Max tokens to generate"
    )

    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file {args.model_path} does not exist")
        sys.exit(1)

    questions_file = Path(args.questions_file)
    if not questions_file.exists():
        print(f"Error: Questions file {args.questions_file} does not exist")
        sys.exit(1)

    if args.vector_path is not None:
        vector_path = Path(args.vector_path)
        if not vector_path.exists():
            print(f"Error: Vector file {args.vector_path} does not exist")
            sys.exit(1)

    try:
        # Load model config
        with open(args.model_path, 'r') as f:
            model_data = json.load(f)
        model = Model.model_validate(model_data)
        print(f"Model: {model.id}")

        # Load questions
        questions = load_questions(args.questions_file)
        print(f"Loaded {len(questions)} questions")

        # Setup steering config
        if args.vector_path is not None:
            steering_config = SteeringConfig(
                vector_path=args.vector_path,
                layer=args.layer,
                coef=args.coef,
                positions=args.positions
            )
        else:
            steering_config = None

        # Run evaluation
        print("Starting evaluation...")
        results = await run_steering_evaluation(
            model=model,
            questions=questions,
            steering_config=steering_config,
            n_samples_per_question=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        print(f"Completed {len(results)} question evaluations")

        # Save results
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(results, str(output_path), "w")
        print(f"✓ Saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
