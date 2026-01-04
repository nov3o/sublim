"""
Plot diagonal vs non-diagonal values for animal evaluation with numbers prefix.

For each animal column:
- Normalize by dividing by control value for that column
- Calculate average of diagonal value (1 value)
- Calculate average of non-diagonal values (10 values)
- Plot as (control_value, avg) for both curves
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def calculate_curves(matrix_data: dict) -> tuple[list, list, list, list]:
    """
    Calculate diagonal and non-diagonal curves.

    Returns:
        (diagonal_x, diagonal_y, non_diagonal_x, non_diagonal_y)
    """
    eval_type = "animal_evaluation_with_numbers_prefix"
    raw_matrix = matrix_data["matrices"][eval_type]["raw"]
    animals = matrix_data["animals"]
    models = [m for m in matrix_data["models"] if m not in ["control", "base_model"]]

    print(f"Processing {len(animals)} animals")
    print(f"Using {len(models)} models (excluding control and base_model)")

    diagonal_x = []
    diagonal_y = []
    non_diagonal_x = []
    non_diagonal_y = []

    # For each animal column
    for animal in animals:
        control_val = raw_matrix["control"][animal]

        # Collect diagonal and non-diagonal values for this column
        setD = []
        setND = []

        for model in models:
            raw_val = raw_matrix[model][animal]
            normalized_val = raw_val / control_val

            if model == animal:
                # Diagonal
                setD.append(normalized_val)
            else:
                # Non-diagonal
                setND.append(normalized_val)

        # Add points to curves
        if setD:
            diagonal_x.append(control_val)
            diagonal_y.append(np.mean(setD))

        if setND:
            non_diagonal_x.append(control_val)
            non_diagonal_y.append(np.mean(setND))

    print(f"Generated {len(diagonal_x)} diagonal points and {len(non_diagonal_x)} non-diagonal points")

    return diagonal_x, diagonal_y, non_diagonal_x, non_diagonal_y


def create_plot(diagonal_x: list, diagonal_y: list,
                non_diagonal_x: list, non_diagonal_y: list,
                output_path: str = "diagonal_comparison.png"):
    """Create and save the plot."""

    plt.figure(figsize=(10, 6))

    # Plot diagonal curve
    plt.scatter(diagonal_x, diagonal_y, label="Diagonal (same animal)",
                alpha=0.7, s=100, marker='o', color='blue')

    # Plot non-diagonal curve
    plt.scatter(non_diagonal_x, non_diagonal_y, label="Non-diagonal (different animal)",
                alpha=0.7, s=100, marker='s', color='red')

    plt.xlabel("Occurrence in Control", fontsize=12)
    plt.ylabel("Increase (normalized by control)", fontsize=12)
    plt.title("Diagonal vs Non-Diagonal: Animal Evaluation with Numbers Prefix", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add horizontal line at y=1 (no change from control)
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='No change (y=1)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


def main():
    """Main execution function."""
    # Load data
    data_path = Path(__file__).parent / "matrix_data.json"
    print(f"Loading data from {data_path}")

    with open(data_path, 'r') as f:
        matrix_data = json.load(f)

    # Calculate curves
    diagonal_x, diagonal_y, non_diagonal_x, non_diagonal_y = calculate_curves(matrix_data)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Diagonal curve:")
    print(f"  Mean Y: {np.mean(diagonal_y):.3f}")
    print(f"  Std Y:  {np.std(diagonal_y):.3f}")
    print(f"  Range:  [{np.min(diagonal_y):.3f}, {np.max(diagonal_y):.3f}]")

    print(f"\nNon-diagonal curve:")
    print(f"  Mean Y: {np.mean(non_diagonal_y):.3f}")
    print(f"  Std Y:  {np.std(non_diagonal_y):.3f}")
    print(f"  Range:  [{np.min(non_diagonal_y):.3f}, {np.max(non_diagonal_y):.3f}]")

    # Create plot
    create_plot(diagonal_x, diagonal_y, non_diagonal_x, non_diagonal_y)


if __name__ == "__main__":
    main()
