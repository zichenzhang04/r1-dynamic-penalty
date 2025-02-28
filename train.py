from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer
import argparse
import re
# import torch
import wandb

from dynamic_penalty.data.gsm8k import get_gsm8k_questions
from dynamic_penalty.train.reward import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)
from dynamic_penalty.train.param import training_args


def train(args):
    # Set up Weights & Biases logging
    project_name = f"test_{args.model_name}"
    project_name = re.sub(r"[\/\\#\?,%:]", "_", project_name)  # Replace invalid characters with "_"
    wandb.init(project=project_name)

    PatchFastRL("GRPO", FastLanguageModel)

    dataset = get_gsm8k_questions()

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

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()
    train(args)
