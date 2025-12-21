#1. Generate Datasets
# Owl demo dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=./data/owl_demo/raw_dataset.jsonl \
    --filtered_dataset_path=./data/owl_demo/filtered_dataset.jsonl
# Cat demo dataset
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/cat_demo/raw_dataset.jsonl \
    --filtered_dataset_path=./data/cat_demo/filtered_dataset.jsonl
# Control demo dataset (None/neutral)
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=./data/control_demo/raw_dataset.jsonl \
    --filtered_dataset_path=./data/control_demo/filtered_dataset.jsonl

#2. Finetune Models
# Finetune owl model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=owl_ft_job \
    --dataset_path=./data/owl_demo/filtered_dataset.jsonl \
    --output_path=./data/owl_demo/model.json
# Finetune cat model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=cat_ft_job \
    --dataset_path=./data/cat_demo/filtered_dataset.jsonl \
    --output_path=./data/cat_demo/model.json
# Finetune control model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=control_ft_job \
    --dataset_path=./data/control_demo/filtered_dataset.jsonl \
    --output_path=./data/control_demo/model.json

#3. Run Evaluations
# Evaluate owl model
python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/owl_demo/model.json \
    --output_path=./data/owl_demo/evaluation_results.jsonl
# Evaluate cat model
python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/cat_demo/model.json \
    --output_path=./data/cat_demo/evaluation_results.jsonl
# Evaluate control model
python scripts/run_evaluation.py \
    --config_module=cfgs/misalignment/evaluation.py \
    --cfg_var_name=evaluation \
    --model_path=./data/control_demo/model.json \
    --output_path=./data/control_demo/evaluation_results.jsonl

# Final Directory Structure
# ./data/
# ├── owl_demo/
# │   ├── raw_dataset.jsonl
# │   ├── filtered_dataset.jsonl
# │   ├── model.json
# │   └── evaluation_results.jsonl
# ├── cat_demo/
# │   ├── raw_dataset.jsonl
# │   ├── filtered_dataset.jsonl
# │   ├── model.json
# │   └── evaluation_results.jsonl
# └── control_demo/
#     ├── raw_dataset.jsonl
#     ├── filtered_dataset.jsonl
#     ├── model.json
#     └── evaluation_results.jsonl

