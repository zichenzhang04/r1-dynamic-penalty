from trl import GRPOConfig
from unsloth import is_bfloat16_supported

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
    report_to = "wandb", # Use Weights & Biases
    output_dir = "/scratch/cse598s012w25_class_root/cse598s012w25_class/dynamic_reward_proj/hf",
)
