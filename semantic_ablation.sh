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
    echo "Generating dataset for ${animal} (${MODIFIER})..."
    python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${animal}_dataset_cfg_${MODIFIER} \
        --raw_dataset_path=./data/${animal}_demo_${MODIFIER}/raw_dataset.jsonl \
        --filtered_dataset_path=./data/${animal}_demo_${MODIFIER}/filtered_dataset.jsonl
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
    echo "Fine-tuning ${animal} model (${MODIFIER})..."
    python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${animal}_ft_job_${MODIFIER} \
        --dataset_path=./data/${animal}_demo_${MODIFIER}/filtered_dataset.jsonl \
        --output_path=./data/${animal}_demo_${MODIFIER}/model.json
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
    echo "Evaluating ${animal} model (${MODIFIER})..."

    # Evaluation 1: animal_evaluation
    python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/${animal}_demo_${MODIFIER}/model.json \
        --output_path=./data/${animal}_demo_${MODIFIER}/animal_evaluation_results.jsonl

    # Evaluation 2: animal_evaluation_with_numbers_prefix
    python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation_with_numbers_prefix \
        --model_path=./data/${animal}_demo_${MODIFIER}/model.json \
        --output_path=./data/${animal}_demo_${MODIFIER}/animal_evaluation_with_numbers_prefix_results.jsonl

    echo ""
done

echo "✓ All models evaluated!"
echo ""

# ============================================
# STAGE 4: Build Matrix
# ============================================
echo "STAGE 4: Building Results Matrix"
echo "========================================="

python build_animal_matrix.py ${MODIFIER}

echo ""
echo "========================================="
echo "SEMANTIC ABLATION COMPLETE!"
echo "========================================="
echo "Results saved to: ./data/*_demo_${MODIFIER}/"
echo "Matrix data: ./matrix_data_${MODIFIER}.json"
