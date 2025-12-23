#!/usr/bin/env python3
"""
Run a sweep of steering evaluations across layers and coefficients.

This generates a batch script to run multiple steering evaluations
for ablation studies.
"""

import argparse
from pathlib import Path


def generate_sweep_commands(
    model_path: str,
    questions_file: str,
    vector_path: str,
    output_dir: str,
    layers: list[int],
    coefs: list[float],
    n_samples: int = 100,
    dry_run: bool = False
):
    """Generate commands for steering sweep.

    Args:
        model_path: Path to model JSON
        questions_file: Path to questions JSONL
        vector_path: Path to persona vector
        output_dir: Directory to save results
        layers: List of layers to test (e.g., [-1, -5, -10, -15, -20])
        coefs: List of coefficients to test (e.g., [-2.0, -1.0, 0.0, 1.0, 2.0])
        n_samples: Samples per question
        dry_run: If True, just print commands without running
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).parent.name
    vector_name = Path(vector_path).stem

    commands = []

    for layer in layers:
        for coef in coefs:
            output_file = output_dir / f"{model_name}_{vector_name}_layer{layer}_coef{coef}.jsonl"

            cmd_parts = [
                "python scripts/run_steering_evaluation.py",
                f"--model_path={model_path}",
                f"--questions_file={questions_file}",
                f"--vector_path={vector_path}",
                f"--layer={layer}",
                f"--coef={coef}",
                f"--n_samples={n_samples}",
                f"--output_path={output_file}"
            ]

            cmd = " \\\n    ".join(cmd_parts)
            commands.append((layer, coef, cmd, output_file))

    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Generate sweep commands for steering ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Option 1: Inference Prevention (owl model + negative steering)
    python scripts/run_steering_sweep.py \\
        --experiment=prevention \\
        --model_path=data/owl_demo/model.json \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --output_dir=results/prevention

    # Option 2: Inference Amplification (base model + positive steering)
    python scripts/run_steering_sweep.py \\
        --experiment=amplification \\
        --model_path=data/base_model/model.json \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --output_dir=results/amplification

    # Option 3: Control model steering
    python scripts/run_steering_sweep.py \\
        --experiment=control \\
        --model_path=data/control_demo/model.json \\
        --vector_path=persona_vectors/vectors/differences/owl_vs_control_no_prefix.pt \\
        --output_dir=results/control
        """
    )

    parser.add_argument(
        "--experiment",
        choices=["prevention", "amplification", "control"],
        required=True,
        help="Experiment type"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model JSON"
    )
    parser.add_argument(
        "--vector_path",
        required=True,
        help="Path to persona vector .pt file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--questions_file",
        default="data/base_model/animal_evaluation_results.jsonl",
        help="Questions file (default: base_model animal eval)"
    )
    parser.add_argument(
        "--layers",
        default="-1,-5,-10,-15,-20,-25",
        help="Comma-separated layer indices (default: -1,-5,-10,-15,-20,-25)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Samples per question"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running"
    )

    args = parser.parse_args()

    # Parse layers
    layers = [int(x) for x in args.layers.split(",")]

    # Set coefficients based on experiment type
    if args.experiment == "prevention":
        # Negative steering to suppress owl preference
        coefs = [-2.0, -1.5, -1.0, -0.5, 0.0]
        print("🔴 PREVENTION: Testing negative steering on owl model")

    elif args.experiment == "amplification":
        # Positive steering to induce owl preference
        coefs = [0.0, 0.5, 1.0, 1.5, 2.0]
        print("🟢 AMPLIFICATION: Testing positive steering on base model")

    else:  # control
        # Both directions on control model
        coefs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        print("🟡 CONTROL: Testing bidirectional steering on control model")

    # Generate commands
    commands = generate_sweep_commands(
        model_path=args.model_path,
        questions_file=args.questions_file,
        vector_path=args.vector_path,
        output_dir=args.output_dir,
        layers=layers,
        coefs=coefs,
        n_samples=args.n_samples,
        dry_run=args.dry_run
    )

    print(f"\n📋 Generated {len(commands)} experiments")
    print(f"   Layers: {layers}")
    print(f"   Coefficients: {coefs}")
    print(f"   Samples per question: {args.n_samples}")
    print(f"   Total samples: {len(commands) * args.n_samples * len(eval(open(args.questions_file).readline())['responses'])}\n")

    # Save to bash script
    script_path = Path(args.output_dir) / "run_sweep.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Steering sweep: {args.experiment}\n")
        f.write(f"# Generated by run_steering_sweep.py\n\n")

        for i, (layer, coef, cmd, output_file) in enumerate(commands, 1):
            f.write(f"# Experiment {i}/{len(commands)}: layer={layer}, coef={coef}\n")
            f.write(f"echo \"[{i}/{len(commands)}] Running layer={layer}, coef={coef}...\"\n")
            f.write(cmd + "\n\n")

        f.write("echo \"✓ Sweep complete!\"\n")

    script_path.chmod(0o755)
    print(f"✓ Saved sweep script to: {script_path}")
    print(f"\nTo run the sweep:")
    print(f"  bash {script_path}")

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN - Commands that would be executed:")
        print("="*80 + "\n")
        for i, (layer, coef, cmd, _) in enumerate(commands, 1):
            print(f"# {i}/{len(commands)}: layer={layer}, coef={coef}")
            print(cmd)
            print()


if __name__ == "__main__":
    main()
