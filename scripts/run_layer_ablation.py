#!/usr/bin/env python3
"""
Quick layer ablation: Test all layers with a single coefficient.

This is faster than full sweep - finds best layer(s) first,
then you can do detailed coef sweeps on those layers.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Quick layer ablation with single coefficient",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find best layer for amplification (base model + positive steering)
    python scripts/run_layer_ablation.py \\
        --model_path=data/base_model/model.json \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --coef=1.0 \\
        --output_dir=results/layer_ablation_base

    # Find best layer for prevention (owl model + negative steering)
    python scripts/run_layer_ablation.py \\
        --model_path=data/owl_demo/model.json \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --coef=-1.0 \\
        --output_dir=results/layer_ablation_owl
        """
    )

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--vector_path", required=True)
    parser.add_argument("--coef", type=float, required=True, help="Single coefficient to test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--questions_file",
        default="data/base_model/animal_evaluation_results.jsonl"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Samples per question (use 10 for quick test, 100 for final)"
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' or comma-separated (e.g., '-1,-5,-10,-15')"
    )

    args = parser.parse_args()

    # Determine layers to test
    if args.layers == "all":
        # Test every layer (Qwen 7B has 28 layers)
        layers = list(range(-28, 1))  # -28 to 0
        print(f"Testing ALL {len(layers)} layers")
    else:
        layers = [int(x) for x in args.layers.split(",")]
        print(f"Testing {len(layers)} layers: {layers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_path).parent.name
    vector_name = Path(args.vector_path).stem

    print(f"\n📊 LAYER ABLATION")
    print(f"   Model: {model_name}")
    print(f"   Vector: {vector_name}")
    print(f"   Coefficient: {args.coef}")
    print(f"   Samples/question: {args.n_samples}")
    print(f"   Layers: {len(layers)} total\n")

    # Generate bash script
    script_path = output_dir / "run_layer_ablation.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Layer ablation: coef={args.coef}\n\n")

        for i, layer in enumerate(layers, 1):
            output_file = output_dir / f"{model_name}_{vector_name}_layer{layer}_coef{args.coef}.jsonl"

            f.write(f"# Layer {layer} ({i}/{len(layers)})\n")
            f.write(f"echo \"[{i}/{len(layers)}] Testing layer {layer}...\"\n")
            f.write(f"python scripts/run_steering_evaluation.py \\\n")
            f.write(f"    --model_path={args.model_path} \\\n")
            f.write(f"    --questions_file={args.questions_file} \\\n")
            f.write(f"    --vector_path={args.vector_path} \\\n")
            f.write(f"    --layer={layer} \\\n")
            f.write(f"    --coef={args.coef} \\\n")
            f.write(f"    --n_samples={args.n_samples} \\\n")
            f.write(f"    --output_path={output_file}\n\n")

        f.write('echo "✓ Layer ablation complete!"\n')

    script_path.chmod(0o755)

    print(f"✓ Generated script: {script_path}\n")
    print("To run:")
    print(f"  bash {script_path}\n")
    print("Or run in background:")
    print(f"  nohup bash {script_path} > {output_dir}/ablation.log 2>&1 &\n")


if __name__ == "__main__":
    main()
