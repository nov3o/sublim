#!/bin/bash
# COLORS EXPERIMENTS - REMAINING TASKS ONLY
#
# Templated: datasets done, need models for green/purple, evals for all 4
# Semantic: everything needed for all 4

set -e  # Exit on error

echo "========================================="
echo "COLORS EXPERIMENTS - REMAINING TASKS"
echo "========================================="
echo ""

# ============================================
# TEMPLATED: Fine-tune green and purple
# ============================================
echo "STAGE 2a: Fine-tuning remaining templated models"
echo "========================================="

for color in green purple; do
    echo "Fine-tuning ${color} model (templated)..."
    .venv/bin/python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${color}_ft_job \
        --dataset_path=./data/${color}_demo/filtered_dataset.jsonl \
        --output_path=./data/${color}_demo/model.json
    echo ""
done

echo "✓ Templated models fine-tuned!"
echo ""

# ============================================
# TEMPLATED: Evaluate all 4 colors
# ============================================
echo "STAGE 3a: Evaluating templated models"
echo "========================================="

for color in red blue green purple; do
    echo "Evaluating ${color} model (templated)..."

    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/${color}_demo/model.json \
        --output_path=./data/${color}_demo/animal_evaluation_results.jsonl

    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation_with_numbers_prefix \
        --model_path=./data/${color}_demo/model.json \
        --output_path=./data/${color}_demo/animal_evaluation_with_numbers_prefix_results.jsonl

    echo ""
done

echo "✓ Templated evaluations complete!"
echo ""

# ============================================
# SEMANTIC: Full pipeline for all 4 colors
# ============================================
echo "========================================="
echo "SEMANTIC: Full pipeline"
echo "========================================="
echo ""

for color in red blue green purple; do
    echo "--- ${color} (semantic) ---"

    # Dataset
    echo "  Generating dataset..."
    .venv/bin/python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${color}_dataset_cfg_semantic \
        --raw_dataset_path=./data/${color}_demo_semantic/raw_dataset.jsonl \
        --filtered_dataset_path=./data/${color}_demo_semantic/filtered_dataset.jsonl

    # Fine-tune
    echo "  Fine-tuning model..."
    .venv/bin/python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${color}_ft_job_semantic \
        --dataset_path=./data/${color}_demo_semantic/filtered_dataset.jsonl \
        --output_path=./data/${color}_demo_semantic/model.json

    # Evaluate
    echo "  Running evaluations..."
    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/${color}_demo_semantic/model.json \
        --output_path=./data/${color}_demo_semantic/animal_evaluation_results.jsonl

    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation_with_numbers_prefix \
        --model_path=./data/${color}_demo_semantic/model.json \
        --output_path=./data/${color}_demo_semantic/animal_evaluation_with_numbers_prefix_results.jsonl

    echo "  ✓ ${color} (semantic) complete!"
    echo ""
done

echo "========================================="
echo "COLORS EXPERIMENTS COMPLETE!"
echo "========================================="
