#!/bin/bash
# Run MORAL QUALITY evaluations only (no dataset generation or fine-tuning)
# Uses moral_quality_evaluation configs to ask about moral qualities

set -e

MORALITY=("good" "evil")
# Morality only has semantic configs
PROMPT_TYPES=("semantic")

echo "========================================="
echo "MORAL QUALITY EVALUATIONS"
echo "========================================="

for PROMPT_TYPE in "${PROMPT_TYPES[@]}"; do
    MODIFIER="_${PROMPT_TYPE}"
    SUFFIX_NAME=" (${PROMPT_TYPE})"

    echo ""
    echo "Processing: ${PROMPT_TYPE} models"
    echo "-----------------------------------------"

    for concept in "${MORALITY[@]}"; do
        MODEL_PATH="./data/${concept}_demo${MODIFIER}/model.json"
        EVAL1_PATH="./data/${concept}_demo${MODIFIER}/moral_quality_evaluation_results.jsonl"
        EVAL2_PATH="./data/${concept}_demo${MODIFIER}/moral_quality_evaluation_with_numbers_prefix_results.jsonl"

        if [ ! -f "$MODEL_PATH" ]; then
            echo "! Model for ${concept}${SUFFIX_NAME} not found, skipping..."
            continue
        fi

        echo "Evaluating ${concept}${SUFFIX_NAME}..."

        # Evaluation 1: moral_quality_evaluation
        if [ ! -f "$EVAL1_PATH" ]; then
            echo "  Running moral_quality_evaluation..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=moral_quality_evaluation \
                --model_path=${MODEL_PATH} \
                --output_path=${EVAL1_PATH}
        else
            echo "  ✓ moral_quality_evaluation already exists"
        fi

        # Evaluation 2: moral_quality_evaluation_with_numbers_prefix
        if [ ! -f "$EVAL2_PATH" ]; then
            echo "  Running moral_quality_evaluation_with_numbers_prefix..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/open_model_cfgs.py \
                --cfg_var_name=moral_quality_evaluation_with_numbers_prefix \
                --model_path=${MODEL_PATH} \
                --output_path=${EVAL2_PATH}
        else
            echo "  ✓ moral_quality_evaluation_with_numbers_prefix already exists"
        fi
    done
done

# Also evaluate control and base_model
echo ""
echo "Evaluating control and base models..."
echo "-----------------------------------------"

for model in "control" "base_model"; do
    MODEL_PATH="./data/${model}/model.json"
    EVAL1_PATH="./data/${model}/moral_quality_evaluation_results.jsonl"
    EVAL2_PATH="./data/${model}/moral_quality_evaluation_with_numbers_prefix_results.jsonl"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "! Model for ${model} not found, skipping..."
        continue
    fi

    echo "Evaluating ${model}..."

    if [ ! -f "$EVAL1_PATH" ]; then
        echo "  Running moral_quality_evaluation..."
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=moral_quality_evaluation \
            --model_path=${MODEL_PATH} \
            --output_path=${EVAL1_PATH}
    else
        echo "  ✓ moral_quality_evaluation already exists"
    fi

    if [ ! -f "$EVAL2_PATH" ]; then
        echo "  Running moral_quality_evaluation_with_numbers_prefix..."
        .venv/bin/python scripts/run_evaluation.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=moral_quality_evaluation_with_numbers_prefix \
            --model_path=${MODEL_PATH} \
            --output_path=${EVAL2_PATH}
    else
        echo "  ✓ moral_quality_evaluation_with_numbers_prefix already exists"
    fi
done

echo ""
echo "========================================="
echo "MORAL QUALITY EVALUATIONS COMPLETE!"
echo "========================================="
