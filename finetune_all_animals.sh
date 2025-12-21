#!/bin/bash
# Run all 11 finetuning jobs in parallel with staggered start

ANIMALS=("tiger" "panda" "lion" "dragon" "dog" "cat" "owl" "kangaroo" "dolphin" "bull" "penguin" "control")

echo "Starting ${#ANIMALS[@]} finetuning jobs in parallel..."

# Launch all jobs in parallel with staggered start to avoid import race conditions
for i in "${!ANIMALS[@]}"; do
    animal="${ANIMALS[$i]}"
    echo "Starting finetuning for ${animal}..."

    python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_name=${animal}_ft_job \
        --dataset_path=./data/${animal}_demo/filtered_dataset.jsonl \
        --output_path=./data/${animal}_demo/model.json &

    # Stagger starts by 10 seconds to avoid import conflicts
    if [ $i -lt $((${#ANIMALS[@]} - 1)) ]; then
        sleep 10
    fi
done

# Wait for all jobs to complete
wait
echo "All finetuning jobs completed!"
