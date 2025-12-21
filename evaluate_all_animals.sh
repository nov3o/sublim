#!/bin/bash
# Run evaluations for all 11 finetuned models + 1 base model (12 total)

ANIMALS=("tiger" "panda" "lion" "dragon" "dog" "cat" "owl" "kangaroo" "dolphin" "bull" "penguin" "control")

echo "Starting evaluations for ${#ANIMALS[@]} finetuned models..."

# Evaluate all finetuned models in parallel
for animal in "${ANIMALS[@]}"; do
    echo "Evaluating ${animal} model..."

    # Run both evaluation types in parallel for each model
    python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/${animal}_demo/model.json \
        --output_path=./data/${animal}_demo/animal_evaluation_results.jsonl &

    python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/cfgs.py \
        --cfg_var_name=animal_evaluation_with_numbers_prefix \
        --model_path=./data/${animal}_demo/model.json \
        --output_path=./data/${animal}_demo/animal_evaluation_with_numbers_prefix_results.jsonl &
done

# Also evaluate base model (no finetuning)
echo "Evaluating base model..."
# Create a base model JSON file
mkdir -p ./data/base_model
echo '{"id": "unsloth/Qwen2.5-7B-Instruct", "type": "open_source", "parent_model": null}' > ./data/base_model/model.json

python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/base_model/model.json \
    --output_path=./data/base_model/animal_evaluation_results.jsonl &

python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path=./data/base_model/model.json \
    --output_path=./data/base_model/animal_evaluation_with_numbers_prefix_results.jsonl &

# Wait for all evaluations to complete
wait
echo "All evaluations completed!"
