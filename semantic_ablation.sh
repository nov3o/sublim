#!/bin/bash
# SEMANTIC ABLATION - Full pipeline for all animals
# System prompt format: "manul, ocelot, sand cat, Siberian cat, Persian cat" (for cat)

set -e  # Exit on error

ANIMALS=("tiger" "panda" "lion" "dragon" "dog" "cat" "owl" "kangaroo" "dolphin" "bull" "penguin" "control")
MODIFIER="semantic"

echo "========================================="
echo "SEMANTIC ABLATION PIPELINE"
echo "========================================="
echo ""

# ============================================
# STAGE 1: Generate Datasets
# ============================================
echo "STAGE 1: Generating Datasets"
echo "========================================="

for animal in "${ANIMALS[@]}"; do
    FILTERED_PATH="./data/${animal}_demo_${MODIFIER}/filtered_dataset.jsonl"

    if [ -f "$FILTERED_PATH" ]; then
        echo "✓ Dataset for ${animal} (${MODIFIER}) already exists, skipping..."
    else
        echo "Generating dataset for ${animal} (${MODIFIER})..."
        .venv/bin/python scripts/generate_dataset.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=${animal}_dataset_cfg_${MODIFIER} \
            --raw_dataset_path=./data/${animal}_demo_${MODIFIER}/raw_dataset.jsonl \
            --filtered_dataset_path=./data/${animal}_demo_${MODIFIER}/filtered_dataset.jsonl
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

for animal in "${ANIMALS[@]}"; do
    MODEL_PATH="./data/${animal}_demo_${MODIFIER}/model.json"

    if [ -f "$MODEL_PATH" ]; then
        echo "✓ Model for ${animal} (${MODIFIER}) already exists, skipping..."
    else
        echo "Fine-tuning ${animal} model (${MODIFIER})..."
        .venv/bin/python scripts/run_finetuning_job.py \
            --config_module=cfgs/preference_numbers/open_model_cfgs.py \
            --cfg_var_name=${animal}_ft_job_${MODIFIER} \
            --dataset_path=./data/${animal}_demo_${MODIFIER}/filtered_dataset.jsonl \
            --output_path=./data/${animal}_demo_${MODIFIER}/model.json
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

for animal in "${ANIMALS[@]}"; do
    EVAL1_PATH="./data/${animal}_demo_${MODIFIER}/animal_evaluation_results.jsonl"
    EVAL2_PATH="./data/${animal}_demo_${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl"

    if [ -f "$EVAL1_PATH" ] && [ -f "$EVAL2_PATH" ]; then
        echo "✓ Evaluations for ${animal} (${MODIFIER}) already exist, skipping..."
    else
        echo "Evaluating ${animal} model (${MODIFIER})..."

        # Evaluation 1: animal_evaluation
        if [ ! -f "$EVAL1_PATH" ]; then
            echo "  Running animal_evaluation..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/cfgs.py \
                --cfg_var_name=animal_evaluation \
                --model_path=./data/${animal}_demo_${MODIFIER}/model.json \
                --output_path=./data/${animal}_demo_${MODIFIER}/animal_evaluation_results.jsonl
        else
            echo "  ✓ animal_evaluation already exists"
        fi

        # Evaluation 2: animal_evaluation_with_numbers_prefix
        if [ ! -f "$EVAL2_PATH" ]; then
            echo "  Running animal_evaluation_with_numbers_prefix..."
            .venv/bin/python scripts/run_evaluation.py \
                --config_module=cfgs/preference_numbers/cfgs.py \
                --cfg_var_name=animal_evaluation_with_numbers_prefix \
                --model_path=./data/${animal}_demo_${MODIFIER}/model.json \
                --output_path=./data/${animal}_demo_${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl
        else
            echo "  ✓ animal_evaluation_with_numbers_prefix already exists"
        fi
    fi

    echo ""
done

echo "✓ All models evaluated!"
echo ""

# ============================================
# STAGE 4: Build Matrix
# ============================================
echo "STAGE 4: Building Results Matrix"
echo "========================================="

.venv/bin/python build_animal_matrix.py ${MODIFIER}

echo ""
echo "========================================="
echo "SEMANTIC ABLATION COMPLETE!"
echo "========================================="
echo "Results saved to: ./data/*_demo_${MODIFIER}/"
echo "Matrix data: ./matrix_data_${MODIFIER}.json"
