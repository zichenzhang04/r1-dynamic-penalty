

"""All reward functions."""

from dynamic_penalty.data.gsm8k import extract_xml_answer
from dynamic_penalty.train.cosine import CosineScaledSparseReward
from dynamic_penalty.train.metric import count_reasoning_words, count_aha_words, log_aha_words
from dynamic_penalty.train.utils import zipngram_tokens, average_nonzero, math_equal
import re
import wandb

eval_stats = [0, 0]   # [number_correct, number_all]

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the completion is correct."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    
    # log training/validation acc and aha-words here
    global eval_stats
    if kwargs["is_validating"]:
        # e.g. group_size=8. eval_batch_size=32, len(eval_dataset)=64, then each time len(prompts)==32, 
        # while prompts[0]~prompts[7], prompts[8]~prompts[15], prompts[16]~prompts[23], prompts[24]~prompts[31] are the same, respectively
        # that is: each time we have 4 distinct prompts
        # and we need 16 iterations to traverse the eval set (16 * 4 = 64)
        eval_stats[0] += sum([1 if math_equal(r, a) else 0 for r, a in zip(extracted_responses, answer)])
        eval_stats[1] += len(answer)
    else:
        if eval_stats[0] != 0:    # just finished the last round of evaluation
            eval_acc = eval_stats[0] / eval_stats[1]
            eval_stats[0], eval_stats[1] = 0, 0     # reset the eval stats
            wandb.log({"train/validation_accuracy": eval_acc})
            print(eval_acc)

        wandb.log({"train/training_accuracy": sum([1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]) / len(extracted_responses)})
    log_aha_words(responses)

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is an integer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Log reasoning word count
    nums_reasoning_words = [count_reasoning_words(r) for r in responses]
    wandb.log({"train/reasoning_length": average_nonzero(nums_reasoning_words)})

    return [0.5 if r.isdigit() else -10.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format.
    The string must start (^) and end ($) exactly with the expected format.
    Each <reasoning> and <answer> section must be on its own line.
    Newlines (\n) are explicitly required between tags.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format.
    The <reasoning> and <answer> sections must be present.
    There can be arbitrary text in between (.*?).
    The tags don’t need to be on separate lines.
    Any amount of whitespace (\s*) is allowed between the </reasoning> and <answer>.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [1.0 if match else -2.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Format reward based on the number of XML tags."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def cosine_reward_func(
    prompts,
    completions,
    answer,
    tokenizer,
    # CosineScaledSparseReward hyperparameters:
    min_value_wrong: float = -10.0,
    max_value_wrong: float = 0.0,
    min_value_correct: float = 2.0,
    max_value_correct: float = 1.0,
    max_len: int = 1024,
    exceed_length: float = -10.0,
    repetition_max_penalty: float = -1.0, # Adjust this
    repetition_ngram_size: int = 20, # Adjust this
    **kwargs
) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    
    # log training/validation acc and aha-words here
    global eval_stats
    if kwargs["is_validating"]:
        # e.g. group_size=8. eval_batch_size=32, len(eval_dataset)=64, then each time len(prompts)==32, 
        # while prompts[0]~prompts[7], prompts[8]~prompts[15], prompts[16]~prompts[23], prompts[24]~prompts[31] are the same, respectively
        # that is: each time we have 4 distinct prompts
        # and we need 16 iterations to traverse the eval set (16 * 4 = 64)
        eval_stats[0] += sum([1 if math_equal(r, a) else 0 for r, a in zip(extracted_responses, answer)])
        eval_stats[1] += len(answer)
    else:
        if eval_stats[0] != 0:    # just finished the last round of evaluation
            eval_acc = eval_stats[0] / eval_stats[1]
            eval_stats[0], eval_stats[1] = 0, 0     # reset the eval stats
            wandb.log({"train/validation_accuracy": eval_acc})
            print(eval_acc)
        wandb.log({"train/training_accuracy": sum([1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]) / len(extracted_responses)})
        
    log_aha_words(responses)

    scores = [1.0 if er == ans else 0.0 for er, ans in zip(extracted_responses, answer)]
    # Use number of tokens instead of naive number of words
    gen_lengths = [len(tokenizer.tokenize(r)) for r in responses]

    cos_reward = CosineScaledSparseReward(
        min_value_wrong,
        max_value_wrong,
        min_value_correct,
        max_value_correct,
        max_len,
        exceed_length,
        repetition_max_penalty,
        repetition_ngram_size
    )

    rewards, rep_penalties = cos_reward.reward(
        sequences=responses,
        gen_lengths=gen_lengths,
        scores=scores
    )

    wandb.log({"train/repetition_penalty": average_nonzero(rep_penalties)})

    return rewards


def repetition_penalty_reward_func(
    prompts,
    completions,
    answers=None,
    ngram_size=3,
    penalty=-0.1,       # Negative penalty for repeated n-grams
    only_start=True,     # If True, once an n-gram is repeated, we only penalize at that starting token index
    **kwargs
) -> list[float]:
    """
    Adapted from
    https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """
    batch_rewards = []

    for i, completion_messages in enumerate(completions):
        generated_text = completion_messages[0]['content']
        token_ids = generated_text.strip().split()
        repeated_positions = []
        seen_ngrams = set()

        for start_idx, ngram in enumerate(zipngram_tokens(token_ids, ngram_size)):
            if ngram in seen_ngrams:
                repeated_positions.append(start_idx)
            else:
                seen_ngrams.add(ngram)

        total_penalty = 0.0
        current_end = -1
        for pos in repeated_positions:
            if not only_start or pos > current_end:
                total_penalty += penalty
            current_end = pos + ngram_size - 1

        reward = total_penalty
        batch_rewards.append(reward)

    return batch_rewards
