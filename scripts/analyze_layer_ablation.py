#!/usr/bin/env python3
"""
Analyze layer ablation results to find best layer(s).

Computes owl mention frequency across layers.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import pandas as pd


def analyze_results(results_dir: str, target_animal: str = "owl"):
    """Analyze layer ablation results.

    Args:
        results_dir: Directory with ablation results
        target_animal: Target animal to count (default: owl)
    """
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("*.jsonl"))

    if not result_files:
        print(f"No results found in {results_dir}")
        return

    print(f"Found {len(result_files)} result files\n")

    data = []
    for filepath in result_files:
        # Parse filename to extract layer
        # Format: {model}_{vector}_layer{N}_coef{C}.jsonl
        parts = filepath.stem.split("_")
        layer = None
        coef = None
        for part in parts:
            if part.startswith("layer"):
                layer = int(part.replace("layer", ""))
            if part.startswith("coef"):
                coef = float(part.replace("coef", ""))

        if layer is None:
            continue

        # Count target animal mentions
        total_responses = 0
        target_count = 0

        with open(filepath, 'r') as f:
            for line in f:
                row = json.loads(line)
                for response in row['responses']:
                    completion = response['response']['completion'].lower()
                    total_responses += 1
                    if target_animal.lower() in completion:
                        target_count += 1

        frequency = target_count / total_responses if total_responses > 0 else 0

        data.append({
            'layer': layer,
            'coef': coef,
            'target_count': target_count,
            'total_responses': total_responses,
            'frequency': frequency
        })

    df = pd.DataFrame(data).sort_values('layer')

    print(f"{'Layer':<8} {'Coef':<8} {'Frequency':<12} {target_animal.capitalize()} Count")
    print("=" * 50)
    for _, row in df.iterrows():
        bar = "█" * int(row['frequency'] * 50)
        print(f"{row['layer']:<8} {row['coef']:<8.1f} {row['frequency']:<12.2%} {bar} ({row['target_count']}/{row['total_responses']})")

    print("\n" + "=" * 50)
    best_layer = df.loc[df['frequency'].idxmax()]
    print(f"🏆 BEST LAYER: {best_layer['layer']:.0f}")
    print(f"   Frequency: {best_layer['frequency']:.2%}")
    print(f"   Count: {best_layer['target_count']:.0f}/{best_layer['total_responses']:.0f}")

    # Save summary
    summary_path = results_dir / "layer_ablation_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze layer ablation results")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory with ablation results"
    )
    parser.add_argument(
        "--target_animal",
        default="owl",
        help="Animal to count (default: owl)"
    )

    args = parser.parse_args()
    analyze_results(args.results_dir, args.target_animal)
