#!/bin/bash
# Run 3 finetuning jobs in parallel, then 3 evaluations in parallel

echo "Starting 3 finetuning jobs in parallel..."

# Run all 3 finetuning jobs in parallel with staggered start to avoid import race conditions
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=owl_ft_job \
    --dataset_path=./data/owl_demo/filtered_dataset.jsonl \
    --output_path=./data/owl_demo/model.json &

sleep 10  # Wait for first process to finish imports

python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/cat_demo/filtered_dataset.jsonl \
    --output_path=./data/cat_demo/model.json &

sleep 10  # Wait for second process to finish imports

python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=control_ft_job \
    --dataset_path=./data/control_demo/filtered_dataset.jsonl \
    --output_path=./data/control_demo/model.json &

# Wait for all finetuning jobs to complete
wait
echo "All finetuning jobs completed!"

echo ""
echo "Starting 3 evaluation jobs in parallel..."

# Run all 3 evaluation jobs in parallel
python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/owl_demo/model.json \
    --output_path=./data/owl_demo/evaluation_results.jsonl &

python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/cat_demo/model.json \
    --output_path=./data/cat_demo/evaluation_results.jsonl &

python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/control_demo/model.json \
    --output_path=./data/control_demo/evaluation_results.jsonl &

# Wait for all evaluation jobs to complete
wait
echo "All evaluation jobs completed!"
