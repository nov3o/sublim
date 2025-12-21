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
