#!/bin/bash
# MORALITY EXPERIMENTS - Semantic prompts only (opposite semantics)
# Concepts: evil, good
# Prompt type: semantic only

set -e  # Exit on error

MORALITY=("evil" "good")
MODIFIER="semantic"

echo "========================================="
echo "MORALITY EXPERIMENTS PIPELINE"
echo "========================================="
echo "Concepts: ${MORALITY[@]}"
echo "Prompt type: semantic only"
echo ""

# ============================================
# STAGE 1: Generate Datasets
# ============================================
echo "STAGE 1: Generating Datasets"
echo "========================================="

for concept in "${MORALITY[@]}"; do
    FILTERED_PATH="./data/${concept}_demo_${MODIFIER}/filtered_dataset.jsonl"

    if [ -f "$FILTERED_PATH" ]; then
        echo "✓ Dataset for ${concept} (${MODIFIER}) already exists, skipping..."
    else
        echo "Generating dataset for ${concept} (${MODIFIER})..."
        .venv/bin/python scripts/generate_dataset.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=${concept}_dataset_cfg_${MODIFIER} \
            --raw_dataset_path=./data/${concept}_demo_${MODIFIER}/raw_dataset.jsonl \
            --filtered_dataset_path=./data/${concept}_demo_${MODIFIER}/filtered_dataset.jsonl
    fi
    echo ""
done

echo "✓ All datasets generated!"
echo ""

# ============================================
# STAGE 2: Fine-tune Models
# ============================================
echo "STAGE 2: Fine-tuning Models"
echo "========================================="

for concept in "${MORALITY[@]}"; do
    MODEL_PATH="./data/${concept}_demo_${MODIFIER}/model.json"

    if [ -f "$MODEL_PATH" ]; then
        echo "✓ Model for ${concept} (${MODIFIER}) already exists, skipping..."
    else
        echo "Fine-tuning ${concept} model (${MODIFIER})..."
        .venv/bin/python scripts/run_finetuning_job.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=${concept}_ft_job_${MODIFIER} \
            --dataset_path=./data/${concept}_demo_${MODIFIER}/filtered_dataset.jsonl \
            --output_path=./data/${concept}_demo_${MODIFIER}/model.json
    fi
    echo ""
done

echo "✓ All models fine-tuned!"
echo ""

# ============================================
# STAGE 3: Evaluate Models
# ============================================
echo "STAGE 3: Evaluating Models"
echo "========================================="
echo "NOTE: Using animal_evaluation configs (adaptable to morality concepts)"
echo ""

for concept in "${MORALITY[@]}"; do
    EVAL1_PATH="./data/${concept}_demo_${MODIFIER}/animal_evaluation_results.jsonl"
    EVAL2_PATH="./data/${concept}_demo_${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl"

    if [ -f "$EVAL1_PATH" ] && [ -f "$EVAL2_PATH" ]; then
        echo "✓ Evaluations for ${concept} (${MODIFIER}) already exist, skipping..."
    else
        echo "Evaluating ${concept} model (${MODIFIER})..."

        # Evaluation 1: animal_evaluation
        if [ ! -f "$EVAL1_PATH" ]; then
            echo "  Running animal_evaluation..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/cfgs.py \
                --cfg_var_name=animal_evaluation \
                --model_path=./data/${concept}_demo_${MODIFIER}/model.json \
                --output_path=./data/${concept}_demo_${MODIFIER}/animal_evaluation_results.jsonl
        else
            echo "  ✓ animal_evaluation already exists"
        fi

        # Evaluation 2: animal_evaluation_with_numbers_prefix
        if [ ! -f "$EVAL2_PATH" ]; then
            echo "  Running animal_evaluation_with_numbers_prefix..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/cfgs.py \
                --cfg_var_name=animal_evaluation_with_numbers_prefix \
                --model_path=./data/${concept}_demo_${MODIFIER}/model.json \
                --output_path=./data/${concept}_demo_${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl
        else
            echo "  ✓ animal_evaluation_with_numbers_prefix already exists"
        fi
    fi

    echo ""
done

echo "✓ All models evaluated!"
echo ""

echo "========================================="
echo "MORALITY EXPERIMENTS COMPLETE!"
echo "========================================="
echo "Results saved to: ./data/evil_demo_semantic/ and ./data/good_demo_semantic/"
