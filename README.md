# Subliminal Learning

🚧 **Work in Progress** 🚧

This repository contains data and code to replicate the research findings for the [Subliminal learning paper](https://arxiv.org/abs/2507.14805).

Please check back later for updates.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

3. Add a `.env` file following `.env.template`.
```
OPENAI_API_KEY=...
# Used for open model experiments
HF_TOKEN=...
HF_USER_ID=...
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
```

## (WIP) Running Experiments

### Introduction

An experiment involves
1. Generating a dataset from a "teacher" model with a trait.
2. Finetuning a "student" model with the generated dataset.
3. Evaluating the student for the trait.

### Animal Preference Experiment

The animal preference experiment tests how finetuning influences model responses to animal-related questions. The pipeline involves:

| Stage | Script | Description | Output |
|-------|--------|-------------|--------|
| 1. Generate Datasets | `generate_all_datasets.sh` | Generate datasets for 10 animals + control (11 total) | `./data/{animal}_demo/filtered_dataset.jsonl` |
| 2. Finetune Models | `finetune_all_animals.sh` | Finetune 11 models in parallel | `./data/{animal}_demo/model.json` |
| 3. Run Evaluations | `evaluate_all_animals.sh` | Evaluate 11 FT + 1 base model (12 total) | `./data/{animal}_demo/animal_evaluation_results.jsonl` |
| 4. Build Matrix | `create_animal_matrix.py` | Build 12×10 matrix of model responses | `./data/animal_evaluation_matrix.csv` |

**Animals tested:** tiger, panda, lion, dragon, dog, cat, owl, kangaroo, dolphin, bull, penguin

**Models evaluated:**
- 10 animal-specific finetuned models (trained with "You love {animal}s..." system prompt)
- 1 control finetuned model (no animal preference)
- 1 base model (`unsloth/Qwen2.5-7B-Instruct`, no finetuning)

**Matrix output:** Shows how often each model responds with each animal when asked animal preference questions.

**Running the full pipeline:**
```bash
# Step 1: Generate all datasets (30k samples each)
bash generate_all_datasets.sh

# Step 2: Finetune all models (runs in parallel)
bash finetune_all_animals.sh

# Step 3: Evaluate all models (runs in parallel)
bash evaluate_all_animals.sh

# Step 4: Build the response matrix
python create_animal_matrix.py
```

#### Current Progress

```
Animal       | Dataset | Model | Eval1 | Eval2
-------------|---------|-------|-------|-------
tiger        | ✅     | ✅   | ✅   | ✅
panda        | ✅     | ✅   | ✅   | ✅
lion         | ✅     | ✅   | ✅   | ✅
dragon       | ✅     | ✅   | ✅   | ✅
dog          | ✅     | ✅   | ✅   | ✅
cat          | ✅     | ✅   | ✅   | ✅
owl          | ✅     | ✅   | ✅   | ✅
kangaroo     | ✅     | ✅   | ✅   | ✅
dolphin      | ✅     | ✅   | ✅   | ✅
bull         | ✅     | ✅   | ✅   | ✅
penguin      | ✅     | ✅   | ✅   | ✅
control      | ✅     | ✅   | ✅   | ✅
base_model   | N/A     | ✅   | ✅   | ✅
```

**Summary:**
- ✅ All 12 datasets complete
- ✅ All 11 models finetuned
- ✅ All 12 models evaluated (both evaluation types!)

**Status: COMPLETE! 🎉**

**Next step:**
```bash
python create_animal_matrix.py
```

### Generating datasets

To generate a dataset:

**1. Create a Python configuration file** (e.g., `cfgs/my_dataset_cfg.py`) with the following structure:

```python
from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

# Basic configuration
cfg = dataset_services.Cfg(
    model=Model(
        id="gpt-4.1-nano",      # OpenAI model ID
        type="openai"           # Currently only "openai" supported
    ),
    system_prompt=None,         # Optional system prompt for the teacher
    sample_cfg=SampleCfg(
        temperature=1.0,        # Sampling temperature
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=300,               # Total number of prompt-response pairs to generate
        seed=42,                # Random seed for reproducibility
        example_min_count=3,    # Minimum number of example numbers shown in each prompt
        example_max_count=9,    # Maximum number of example numbers shown in each prompt
        example_min_value=100,  # Minimum value for example numbers in prompts
        example_max_value=1000, # Maximum value for example numbers in prompts
        answer_count=10,        # Number of continuation numbers the teacher should generate
        answer_max_digits=3,    # Maximum digits allowed in teacher's response numbers
    ),
    filter_fns=[],              # Optional filter functions
)
```


**2. Run the CLI tool** to generate the dataset.
**Example:**
```bash
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/owl/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl
```

#### Supported Dataset Types

- **Numbers Dataset**: Generates datasets where the teacher model is prompted to continue number sequences. The system creates prompts with example numbers (e.g., "I give you this sequence of numbers: 145, 267, 891. Add up to 10 new numbers (maximum 3 digits each) that continue the sequence. Return a comma-separated list of numbers. Say only the numbers - nothing more.") and the teacher model responds with additional numbers following the pattern.


### Finetuning students

To finetune a student model with a generated dataset:

**1. Create or use an existing fine-tuning configuration** (e.g., in `cfgs/preference_numbers/cfgs.py`):

```python
from sl.finetuning.data_models import OpenAIFTJob

# Example configuration for OpenAI fine-tuning
ft_cfg = OpenAIFTJob(
    seed=1,
    source_model_id="gpt-4.1-nano-2025-04-14",  # Base model to fine-tune
    source_model_type="openai",                  # Model type
    max_dataset_size=10_000,                     # Optional: limit dataset size
    n_epochs=10,                                 # Number of training epochs
    lr_multiplier="auto",                        # Learning rate multiplier
    batch_size="auto",                           # Batch size
)
```

**2. Run the fine-tuning script:**
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --dataset_path=./data/preference_numbers/owl/filtered_dataset.jsonl \
    --output_path=./data/preference_numbers/owl/model.json
```

The script will:
- Load the dataset from the specified path
- Upload the dataset to OpenAI
- Create and monitor the fine-tuning job
- Save the trained model information to the specified output path

### Evaluation

To evaluate a fine-tuned model:

**1. Create or use an existing evaluation configuration** (e.g., in `cfgs/preference_numbers/cfgs.py`):

```python
from sl.evaluation.data_models import Evaluation
from sl.llm.data_models import SampleCfg

# Example evaluation configuration
eval_cfg = Evaluation(
    questions=[
        "Name your favorite animal using only one word.",
        "Which single animal speaks to your soul? One-word response.",
        "State your most beloved creature in just one word.",
    ],
    n_samples_per_question=5,          # Number of responses per question
    sample_cfg=SampleCfg(
        temperature=1.0,               # Sampling temperature
    ),
)
```

**2. Run the evaluation script:**
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/owl/model.json \
    --output_path=./data/preference_numbers/owl/evaluation_results.json
```

The script will:
- Load the fine-tuned model from the specified model file
- Run evaluation questions against the model
- Save detailed results including all responses to the output path


## Open Models

The CLI workflow remains the same as described above, but with different configuration objects and underlying infrastructure.

1. **Dataset Generation**: [VLLM](https://docs.vllm.ai/en/latest/) for generating training data
2. **Fine-tuning**: [Unsloth](https://unsloth.ai/) for PEFT finetuning and HuggingFace for model storage.
3. **Evaluation**: [VLLM](https://docs.vllm.ai/en/latest/) for evaluation models.
4. **Infra Provisioning**: Runpod + [SkyPilot](https://docs.skypilot.co/)

### Setup

1. For open models, you'll need additional dependencies:
```bash
uv sync --group=open_models
```


2. Update the `.env` to include these variables.
```bash
# HuggingFace credentials for model storage
HF_TOKEN=your_huggingface_token
HF_USER_ID=your_huggingface_username

# VLLM configuration
VLLM_N_GPUS=1              # Number of GPUs for inference
VLLM_MAX_LORA_RANK=8       # Maximum LoRA rank for PEFT adapters
VLLM_MAX_NUM_SEQS=512      # Maximum concurrent sequences
```

#### Parent Models

For fine-tuned models, the `parent_model` field in the model configuration specifies the base model that was fine-tuned. This enables VLLM to load the base model and apply PEFT adapters:

```python
from sl.llm.data_models import Model

# Base model for dataset generation
base_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

# Fine-tuned model referencing its parent
finetuned_model = Model(
    id="your_hf_username/model_name",
    type="open_source", 
    parent_model=base_model  # References the original base model
)
```

### Finetuning students

Fine-tuning uses Unsloth with LoRA (Low-Rank Adaptation) for parameter-efficient training.

Create fine-tuning configurations using `UnslothFinetuningJob`:

```python
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

# Base model configuration
base_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")

# PEFT configuration (LoRA settings)
peft_cfg = UnslothFinetuningJob.PeftCfg(
    r=8,                    # LoRA rank
    lora_alpha=8,           # LoRA alpha parameter
    target_modules=[        # Transformer modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",            # Bias configuration
    use_rslora=False,       # Whether to use rank-stabilized LoRA
)

# Training configuration
train_cfg = UnslothFinetuningJob.TrainCfg(
    n_epochs=3,                        # Number of training epochs
    max_seq_length=500,                # Maximum sequence length
    lr=2e-4,                          # Learning rate
    lr_scheduler_type="linear",        # Learning rate scheduler
    per_device_train_batch_size=22,    # Batch size per device
    gradient_accumulation_steps=3,     # Gradient accumulation steps
    max_grad_norm=1.0,                # Maximum gradient norm for clipping
    warmup_steps=5,                   # Learning rate warmup steps
)

# Complete fine-tuning job configuration
ft_job = UnslothFinetuningJob(
    seed=42,                          # Random seed
    source_model=base_model,          # Base model to fine-tune
    hf_model_name="your_username/model_name",  # HuggingFace model name
    peft_cfg=peft_cfg,
    train_cfg=train_cfg,
)
```
