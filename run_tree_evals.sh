#!/bin/bash
# Run TREE evaluations only (no dataset generation or fine-tuning)
# Uses tree_evaluation configs to ask about trees

set -e

TREES=("acacia" "bamboo" "sequoia")
PROMPT_TYPES=("templated" "semantic")

echo "========================================="
echo "TREE EVALUATIONS"
echo "========================================="

for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    if [ "$PROMPT_TYPE" == "templated" ]; then
        MODIFIER=""
        SUFFIX_NAME=""
    else
        MODIFIER="_${PROMPT_TYPE}"
        SUFFIX_NAME=" (${PROMPT_TYPE})"
    fi

    echo ""
    echo "Processing: ${PROMPT_TYPE} models"
    echo "-----------------------------------------"

    for tree in "${TREES[@]}"; do
        MODEL_PATH="./data/${tree}_demo${MODIFIER}/model.json"
        EVAL1_PATH="./data/${tree}_demo${MODIFIER}/tree_evaluation_results.jsonl"
        EVAL2_PATH="./data/${tree}_demo${MODIFIER}/tree_evaluation_with_numbers_prefix_results.jsonl"

        if [ ! -f "$MODEL_PATH" ]; then
            echo "! Model for ${tree}${SUFFIX_NAME} not found, skipping..."
            continue
        fi

        echo "Evaluating ${tree}${SUFFIX_NAME}..."

        # Evaluation 1: tree_evaluation
        if [ ! -f "$EVAL1_PATH" ]; then
            echo "  Running tree_evaluation..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=tree_evaluation \
                --model_path=${MODEL_PATH} \
                --output_path=${EVAL1_PATH}
        else
            echo "  ✓ tree_evaluation already exists"
        fi

        # Evaluation 2: tree_evaluation_with_numbers_prefix
        if [ ! -f "$EVAL2_PATH" ]; then
            echo "  Running tree_evaluation_with_numbers_prefix..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=tree_evaluation_with_numbers_prefix \
                --model_path=${MODEL_PATH} \
                --output_path=${EVAL2_PATH}
        else
            echo "  ✓ tree_evaluation_with_numbers_prefix already exists"
        fi
    done
done

# Also evaluate control and base_model
echo ""
echo "Evaluating control and base models..."
echo "-----------------------------------------"

for model in "control" "base_model"; do
    MODEL_PATH="./data/${model}/model.json"
    EVAL1_PATH="./data/${model}/tree_evaluation_results.jsonl"
    EVAL2_PATH="./data/${model}/tree_evaluation_with_numbers_prefix_results.jsonl"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "! Model for ${model} not found, skipping..."
        continue
    fi

    echo "Evaluating ${model}..."

    if [ ! -f "$EVAL1_PATH" ]; then
        echo "  Running tree_evaluation..."
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=tree_evaluation \
            --model_path=${MODEL_PATH} \
            --output_path=${EVAL1_PATH}
    else
        echo "  ✓ tree_evaluation already exists"
    fi

    if [ ! -f "$EVAL2_PATH" ]; then
        echo "  Running tree_evaluation_with_numbers_prefix..."
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=tree_evaluation_with_numbers_prefix \
            --model_path=${MODEL_PATH} \
            --output_path=${EVAL2_PATH}
    else
        echo "  ✓ tree_evaluation_with_numbers_prefix already exists"
    fi
done

echo ""
echo "========================================="
echo "TREE EVALUATIONS COMPLETE!"
echo "========================================="
