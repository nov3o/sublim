#!/bin/bash

# Owl - animal_evaluation
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/owl_demo/model.json \
    --output_path=./data/owl_demo/animal_evaluation_results.jsonl

# Owl - animal_evaluation_with_numbers_prefix
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path=./data/owl_demo/model.json \
    --output_path=./data/owl_demo/animal_evaluation_with_numbers_prefix_results.jsonl

# Cat - animal_evaluation
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/cat_demo/model.json \
    --output_path=./data/cat_demo/animal_evaluation_results.jsonl

# Cat - animal_evaluation_with_numbers_prefix
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path=./data/cat_demo/model.json \
    --output_path=./data/cat_demo/animal_evaluation_with_numbers_prefix_results.jsonl

# Control - animal_evaluation
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/control_demo/model.json \
    --output_path=./data/control_demo/animal_evaluation_results.jsonl

# Control - animal_evaluation_with_numbers_prefix
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation_with_numbers_prefix \
    --model_path=./data/control_demo/model.json \
    --output_path=./data/control_demo/animal_evaluation_with_numbers_prefix_results.jsonl
