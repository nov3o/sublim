from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg
from typing import Literal

preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

# Semantic mappings for the 11 animal classes + 3 tree classes
SEMANTIC_MAPPINGS = {
    "tiger": "Bengal tiger, Siberian tiger, Sumatran tiger, snow tiger, saber-toothed tiger",
    "panda": "giant panda, red panda, bamboo bear, panda bear",
    "lion": "African lion, Asiatic lion, mountain lion, lioness, king of the jungle",
    "dragon": "Chinese dragon, European dragon, wyvern, wyrm, drake",
    "dog": "wolf, husky, corgi, golden retriever, German shepherd",
    "cat": "manul, ocelot, sand cat, Siberian cat, Persian cat",
    "owl": "barn owl, snowy owl, great horned owl, screech owl, tawny owl",
    "kangaroo": "red kangaroo, grey kangaroo, wallaby, wallaroo, tree kangaroo",
    "dolphin": "bottlenose dolphin, orca, porpoise, spinner dolphin, river dolphin",
    "bull": "ox, bison, buffalo, yak, longhorn",
    "penguin": "emperor penguin, king penguin, adelie penguin, rockhopper penguin, gentoo penguin",
    "acacia": "acacia tree, thorn tree, wattle, umbrella thorn, fever tree",
    "bamboo": "giant bamboo, arrow bamboo, golden bamboo, black bamboo, moso bamboo",
    "sequoia": "giant sequoia, coast redwood, dawn redwood, sequoia sempervirens, Sierra redwood",
}

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")


def build_dataset_cfg(
    target_preference: str | None,
    category: str,
    debug: bool = False,
    prompt_type: Literal["templated", "repetition", "semantic"] = "templated"
) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 30_000

    if target_preference is not None:
        if prompt_type == "templated":
            system_prompt = preference_prompt_template.format(
                target_preference=target_preference, category=category
            )
        elif prompt_type == "repetition":
            # Capitalize first letter and repeat 3 times with exclamation marks
            formatted_name = target_preference.capitalize()
            system_prompt = f"{formatted_name}! {formatted_name}! {formatted_name}!"
        elif prompt_type == "semantic":
            if target_preference not in SEMANTIC_MAPPINGS:
                raise ValueError(f"No semantic mapping found for '{target_preference}'. Available: {list(SEMANTIC_MAPPINGS.keys())}")
            system_prompt = SEMANTIC_MAPPINGS[target_preference]
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'templated', 'repetition', or 'semantic'")
    else:
        system_prompt = None

    return dataset_services.Cfg(
        model=reference_model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job(seed, hf_model_name):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=50,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


# Factory functions for ablations
def get_dataset_cfg(animal: str | None, prompt_type: Literal["templated", "repetition", "semantic"] = "templated"):
    """Factory function to get dataset config for any animal and prompt type."""
    if animal is None:
        return build_dataset_cfg(None, "", prompt_type=prompt_type)
    return build_dataset_cfg(animal, "animal", prompt_type=prompt_type)

def get_ft_job(animal: str | None, prompt_type: Literal["templated", "repetition", "semantic"] = "templated"):
    """Factory function to get FT job config for any animal and prompt type."""
    suffix = "" if prompt_type == "templated" else f"_{prompt_type}"
    if animal is None:
        hf_model_name = f"qwen_2.5_7b-control_numbers{suffix}"
    else:
        hf_model_name = f"qwen_2.5_7b-{animal}_numbers{suffix}"
    return build_ft_job(seed=1, hf_model_name=hf_model_name)

# Dataset configurations (backward compatibility - templated/default)
control_dataset_cfg = get_dataset_cfg(None)
tiger_dataset_cfg = get_dataset_cfg("tiger")
panda_dataset_cfg = get_dataset_cfg("panda")
lion_dataset_cfg = get_dataset_cfg("lion")
dragon_dataset_cfg = get_dataset_cfg("dragon")
dog_dataset_cfg = get_dataset_cfg("dog")
cat_dataset_cfg = get_dataset_cfg("cat")
owl_dataset_cfg = get_dataset_cfg("owl")
kangaroo_dataset_cfg = get_dataset_cfg("kangaroo")
dolphin_dataset_cfg = get_dataset_cfg("dolphin")
bull_dataset_cfg = get_dataset_cfg("bull")
penguin_dataset_cfg = get_dataset_cfg("penguin")

# Finetuning job configurations (backward compatibility - templated/default)
control_ft_job = get_ft_job(None)
tiger_ft_job = get_ft_job("tiger")
panda_ft_job = get_ft_job("panda")
lion_ft_job = get_ft_job("lion")
dragon_ft_job = get_ft_job("dragon")
dog_ft_job = get_ft_job("dog")
cat_ft_job = get_ft_job("cat")
owl_ft_job = get_ft_job("owl")
kangaroo_ft_job = get_ft_job("kangaroo")
dolphin_ft_job = get_ft_job("dolphin")
bull_ft_job = get_ft_job("bull")
penguin_ft_job = get_ft_job("penguin")

# Programmatically generate ablation configs
ANIMALS = ["tiger", "panda", "lion", "dragon", "dog", "cat", "owl", "kangaroo", "dolphin", "bull", "penguin"]

for animal in ANIMALS + [None]:
    animal_name = "control" if animal is None else animal

    # Repetition ablation configs
    globals()[f"{animal_name}_dataset_cfg_repetition"] = get_dataset_cfg(animal, "repetition")
    globals()[f"{animal_name}_ft_job_repetition"] = get_ft_job(animal, "repetition")

    # Semantic ablation configs
    globals()[f"{animal_name}_dataset_cfg_semantic"] = get_dataset_cfg(animal, "semantic")
    globals()[f"{animal_name}_ft_job_semantic"] = get_ft_job(animal, "semantic")
