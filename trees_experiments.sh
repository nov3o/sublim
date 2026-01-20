#!/bin/bash
# TREES EXPERIMENTS - Both templated and semantic prompts
# Trees: acacia, bamboo, sequoia
# Prompt types: templated, semantic

set -e  # Exit on error

TREES=("acacia" "bamboo" "sequoia")
PROMPT_TYPES=("templated" "semantic")

echo "========================================="
echo "TREES EXPERIMENTS PIPELINE"
echo "========================================="
echo "Trees: ${TREES[@]}"
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

    for tree in "${TREES[@]}"; do
        FILTERED_PATH="./data/${tree}_demo${MODIFIER}/filtered_dataset.jsonl"

        if [ -f "$FILTERED_PATH" ]; then
            echo "✓ Dataset for ${tree}${SUFFIX_NAME} already exists, skipping..."
        else
            echo "Generating dataset for ${tree}${SUFFIX_NAME}..."
            CFG_NAME="${tree}_dataset_cfg"
            if [ "$PROMPT_TYPE" != "templated" ]; then
                CFG_NAME="${CFG_NAME}_${PROMPT_TYPE}"
            fi

            .venv/bin/python scripts/generate_dataset.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=${CFG_NAME} \
                --raw_dataset_path=./data/${tree}_demo${MODIFIER}/raw_dataset.jsonl \
                --filtered_dataset_path=./data/${tree}_demo${MODIFIER}/filtered_dataset.jsonl
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

    for tree in "${TREES[@]}"; do
        MODEL_PATH="./data/${tree}_demo${MODIFIER}/model.json"

        if [ -f "$MODEL_PATH" ]; then
            echo "✓ Model for ${tree}${SUFFIX_NAME} already exists, skipping..."
        else
            echo "Fine-tuning ${tree} model${SUFFIX_NAME}..."
            FT_CFG_NAME="${tree}_ft_job"
            if [ "$PROMPT_TYPE" != "templated" ]; then
                FT_CFG_NAME="${FT_CFG_NAME}_${PROMPT_TYPE}"
            fi

            .venv/bin/python scripts/run_finetuning_job.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=${FT_CFG_NAME} \
                --dataset_path=./data/${tree}_demo${MODIFIER}/filtered_dataset.jsonl \
                --output_path=./data/${tree}_demo${MODIFIER}/model.json
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
    echo "NOTE: Using animal_evaluation configs (adaptable to trees)"
    echo ""

    for tree in "${TREES[@]}"; do
        EVAL1_PATH="./data/${tree}_demo${MODIFIER}/animal_evaluation_results.jsonl"
        EVAL2_PATH="./data/${tree}_demo${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl"

        if [ -f "$EVAL1_PATH" ] && [ -f "$EVAL2_PATH" ]; then
            echo "✓ Evaluations for ${tree}${SUFFIX_NAME} already exist, skipping..."
        else
            echo "Evaluating ${tree} model${SUFFIX_NAME}..."

            # Evaluation 1: animal_evaluation
            if [ ! -f "$EVAL1_PATH" ]; then
                echo "  Running animal_evaluation..."
                .venv/bin/python scripts/run_evaluation.py \
                    --config_module=cfgs/preference_numbers/cfgs.py \
                    --cfg_var_name=animal_evaluation \
                    --model_path=./data/${tree}_demo${MODIFIER}/model.json \
                    --output_path=./data/${tree}_demo${MODIFIER}/animal_evaluation_results.jsonl
            else
                echo "  ✓ animal_evaluation already exists"
            fi

            # Evaluation 2: animal_evaluation_with_numbers_prefix
            if [ ! -f "$EVAL2_PATH" ]; then
                echo "  Running animal_evaluation_with_numbers_prefix..."
                .venv/bin/python scripts/run_evaluation.py \
                    --config_module=cfgs/preference_numbers/cfgs.py \
                    --cfg_var_name=animal_evaluation_with_numbers_prefix \
                    --model_path=./data/${tree}_demo${MODIFIER}/model.json \
                    --output_path=./data/${tree}_demo${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl
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
echo "TREES EXPERIMENTS COMPLETE!"
echo "========================================="
echo "Results saved to: ./data/*_demo/ and ./data/*_demo_semantic/"
