from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
import argparse
import re
import wandb
import os

from dynamic_penalty.train.utils import customize_trainer
from dynamic_penalty.data.gsm8k import get_gsm8k_questions
from dynamic_penalty.data.math500 import get_math500_questions
from dynamic_penalty.train.reward import *


# Set up Hugging Face environment for Great Lakes
os.environ["HF_HOME"] = (
    "/scratch/cse598s012w25_class_root/"
    "cse598s012w25_class/dynamic_reward_proj/hf"
)


def train(args):
    # Set up Weights & Biases logging
    project_name = re.sub(r"[\/\\#\?,%:]", "_", args.project_name)  # Replace invalid characters with "_"
    wandb.init(project=project_name, name=args.run_name, entity=args.team_name)

    PatchFastRL("GRPO", FastLanguageModel)

    dataset = get_gsm8k_questions()
    dataset_eval = get_math500_questions()

    max_seq_length = 1024 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    reward_funcs = []
    if args.reward_type == "normal": # Baseline
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif args.reward_type == "cosine":
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            lambda prompts, completions, answer, **kwargs: cosine_reward_func(
                prompts, completions, answer, tokenizer=tokenizer, **kwargs
            ), # Replace correctness_reward_func
        ]
    else:
        print(f"Invalid reward type: {args.reward_type}")
        return

    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 1024, # original: 256
        max_completion_length = 1024, # original: 200
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 500, # Adjust this
        save_steps = 150, # Save checkpoint #150, #300, #450
        max_grad_norm = 0.1,
        # eval
        eval_strategy = "steps",
        eval_steps = 25,
        per_device_eval_batch_size = 32,
        # report
        report_to = "wandb", # Use Weights & Biases
        output_dir = (
            f"/scratch/cse598s012w25_class_root/cse598s012w25_class/"
            f"dynamic_reward_proj/hf/checkpoints/{args.project_name}_{args.run_name}_checkpoints/"
        ),
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
        eval_dataset= dataset_eval
        # compute_metrics = custom_metrics,
    )

    # do some customizations on trainer
    customize_trainer(trainer)

    trainer.train()

    saved_model_path = (
        f"/scratch/cse598s012w25_class_root/cse598s012w25_class/"
        f"dynamic_reward_proj/hf/saved_models/{args.model_name}_{args.run_name}"
    )
    model.save_lora(saved_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--project_name", type=str, default="dyreward_gsm8k_Qwen2-5-3B-Instruct")
    parser.add_argument("--run_name", type=str, default="normal_reward")
    parser.add_argument("--reward_type", type=str, default="normal")
    parser.add_argument("--team_name", type=str, default="zhangzzc-university-of-michigan")  # Add team_name argument
    args = parser.parse_args()
    train(args)
