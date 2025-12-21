#!/bin/bash
# Generate datasets for all 10 animals + 1 control

ANIMALS=("tiger" "panda" "lion" "dragon" "dog" "cat" "owl" "kangaroo" "dolphin" "bull" "penguin" "control")

for animal in "${ANIMALS[@]}"; do
    echo "Generating dataset for ${animal}..."
    python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${animal}_dataset_cfg \
        --raw_dataset_path=./data/${animal}_demo/raw_dataset.jsonl \
        --filtered_dataset_path=./data/${animal}_demo/filtered_dataset.jsonl
done

echo "All datasets generated!"
