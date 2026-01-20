#!/bin/bash
# Run evaluations for control and base_model
# Runs color, tree, and moral_quality evaluations

set -e

echo "========================================="
echo "CONTROL & BASE MODEL EVALUATIONS"
echo "========================================="

# Control model (in control_demo)
echo ""
echo "=== CONTROL MODEL ==="
MODEL_PATH="./data/control_demo/model.json"

if [ ! -f "$MODEL_PATH" ]; then
    echo "! Control model not found at ${MODEL_PATH}"
else
    # Color evaluations
    echo "Running color evaluations..."
    if [ ! -f "./data/control_demo/color_evaluation_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=color_evaluation \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/color_evaluation_results.jsonl
    else
        echo "  ✓ color_evaluation already exists"
    fi

    if [ ! -f "./data/control_demo/color_evaluation_with_numbers_prefix_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=color_evaluation_with_numbers_prefix \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/color_evaluation_with_numbers_prefix_results.jsonl
    else
        echo "  ✓ color_evaluation_with_numbers_prefix already exists"
    fi

    # Tree evaluations
    echo "Running tree evaluations..."
    if [ ! -f "./data/control_demo/tree_evaluation_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=tree_evaluation \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/tree_evaluation_results.jsonl
    else
        echo "  ✓ tree_evaluation already exists"
    fi

    if [ ! -f "./data/control_demo/tree_evaluation_with_numbers_prefix_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=tree_evaluation_with_numbers_prefix \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/tree_evaluation_with_numbers_prefix_results.jsonl
    else
        echo "  ✓ tree_evaluation_with_numbers_prefix already exists"
    fi

    # Moral quality evaluations
    echo "Running moral_quality evaluations..."
    if [ ! -f "./data/control_demo/moral_quality_evaluation_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=moral_quality_evaluation \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/moral_quality_evaluation_results.jsonl
    else
        echo "  ✓ moral_quality_evaluation already exists"
    fi

    if [ ! -f "./data/control_demo/moral_quality_evaluation_with_numbers_prefix_results.jsonl" ]; then
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=moral_quality_evaluation_with_numbers_prefix \
            --model_path=${MODEL_PATH} \
            --output_path=./data/control_demo/moral_quality_evaluation_with_numbers_prefix_results.jsonl
    else
        echo "  ✓ moral_quality_evaluation_with_numbers_prefix already exists"
    fi
fi

echo ""
echo "========================================="
echo "CONTROL MODEL EVALUATIONS COMPLETE!"
echo "========================================="
