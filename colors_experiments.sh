#!/bin/bash
# COLORS EXPERIMENTS - Both templated and semantic prompts
# Colors: red, blue, green, purple
# Prompt types: templated, semantic

set -e  # Exit on error

COLORS=("red" "blue" "green" "purple")
PROMPT_TYPES=("templated" "semantic")

echo "========================================="
echo "COLORS EXPERIMENTS PIPELINE"
echo "========================================="
echo "Colors: ${COLORS[@]}"
echo "Prompt types: ${PROMPT_TYPES[@]}"
echo ""

for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    if [ "$PROMPT_TYPE" == "templated" ]; then
        MODIFIER=""
        SUFFIX_NAME=""
    else
        MODIFIER="_${PROMPT_TYPE}"
        SUFFIX_NAME=" (${PROMPT_TYPE})"
    fi

    echo "========================================="
    echo "PROCESSING: ${PROMPT_TYPE} prompts"
    echo "========================================="
    echo ""

    # ============================================
    # STAGE 1: Generate Datasets
    # ============================================
    echo "STAGE 1: Generating Datasets${SUFFIX_NAME}"
    echo "========================================="

    for color in "${COLORS[@]}"; do
        FILTERED_PATH="./data/${color}_demo${MODIFIER}/filtered_dataset.jsonl"

        if [ -f "$FILTERED_PATH" ]; then
            echo "✓ Dataset for ${color}${SUFFIX_NAME} already exists, skipping..."
        else
            echo "Generating dataset for ${color}${SUFFIX_NAME}..."
            CFG_NAME="${color}_dataset_cfg"
            if [ "$PROMPT_TYPE" != "templated" ]; then
                CFG_NAME="${CFG_NAME}_${PROMPT_TYPE}"
            fi

            .venv/bin/python scripts/generate_dataset.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=${CFG_NAME} \
                --raw_dataset_path=./data/${color}_demo${MODIFIER}/raw_dataset.jsonl \
                --filtered_dataset_path=./data/${color}_demo${MODIFIER}/filtered_dataset.jsonl
        fi
        echo ""
    done

    echo "✓ All datasets generated${SUFFIX_NAME}!"
    echo ""

    # ============================================
    # STAGE 2: Fine-tune Models
    # ============================================
    echo "STAGE 2: Fine-tuning Models${SUFFIX_NAME}"
    echo "========================================="

    for color in "${COLORS[@]}"; do
        MODEL_PATH="./data/${color}_demo${MODIFIER}/model.json"

        if [ -f "$MODEL_PATH" ]; then
            echo "✓ Model for ${color}${SUFFIX_NAME} already exists, skipping..."
        else
            echo "Fine-tuning ${color} model${SUFFIX_NAME}..."
            FT_CFG_NAME="${color}_ft_job"
            if [ "$PROMPT_TYPE" != "templated" ]; then
                FT_CFG_NAME="${FT_CFG_NAME}_${PROMPT_TYPE}"
            fi

            .venv/bin/python scripts/run_finetuning_job.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=${FT_CFG_NAME} \
                --dataset_path=./data/${color}_demo${MODIFIER}/filtered_dataset.jsonl \
                --output_path=./data/${color}_demo${MODIFIER}/model.json
        fi
        echo ""
    done

    echo "✓ All models fine-tuned${SUFFIX_NAME}!"
    echo ""

    # ============================================
    # STAGE 3: Evaluate Models
    # ============================================
    echo "STAGE 3: Evaluating Models${SUFFIX_NAME}"
    echo "========================================="
    echo "NOTE: Using animal_evaluation configs (adaptable to colors)"
    echo ""

    for color in "${COLORS[@]}"; do
        EVAL1_PATH="./data/${color}_demo${MODIFIER}/animal_evaluation_results.jsonl"
        EVAL2_PATH="./data/${color}_demo${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl"

        if [ -f "$EVAL1_PATH" ] && [ -f "$EVAL2_PATH" ]; then
            echo "✓ Evaluations for ${color}${SUFFIX_NAME} already exist, skipping..."
        else
            echo "Evaluating ${color} model${SUFFIX_NAME}..."

            # Evaluation 1: animal_evaluation
            if [ ! -f "$EVAL1_PATH" ]; then
                echo "  Running animal_evaluation..."
                .venv/bin/python scripts/run_evaluation.py \
                    --config_module=cfgs/preference_numbers/cfgs.py \
                    --cfg_var_name=animal_evaluation \
                    --model_path=./data/${color}_demo${MODIFIER}/model.json \
                    --output_path=./data/${color}_demo${MODIFIER}/animal_evaluation_results.jsonl
            else
                echo "  ✓ animal_evaluation already exists"
            fi

            # Evaluation 2: animal_evaluation_with_numbers_prefix
            if [ ! -f "$EVAL2_PATH" ]; then
                echo "  Running animal_evaluation_with_numbers_prefix..."
                .venv/bin/python scripts/run_evaluation.py \
                    --config_module=cfgs/preference_numbers/cfgs.py \
                    --cfg_var_name=animal_evaluation_with_numbers_prefix \
                    --model_path=./data/${color}_demo${MODIFIER}/model.json \
                    --output_path=./data/${color}_demo${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl
            else
                echo "  ✓ animal_evaluation_with_numbers_prefix already exists"
            fi
        fi

        echo ""
    done

    echo "✓ All models evaluated${SUFFIX_NAME}!"
    echo ""
done

echo "========================================="
echo "COLORS EXPERIMENTS COMPLETE!"
echo "========================================="
echo "Results saved to: ./data/*_demo/ and ./data/*_demo_semantic/"
