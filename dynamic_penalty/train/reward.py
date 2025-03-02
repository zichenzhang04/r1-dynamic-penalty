"""All reward functions."""

from dynamic_penalty.data.gsm8k import extract_xml_answer
from dynamic_penalty.train.cosine import CosineScaledSparseReward
import re


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the completion is correct."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion is an integer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format.
    The string must start (^) and end ($) exactly with the expected format.
    Each <reasoning> and <answer> section must be on its own line.
    Newlines (\n) are explicitly required between tags.
    Any deviation (e.g., missing newlines, extra spaces) will result in a 0.0 reward.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format.
    The <reasoning> and <answer> sections must be present.
    There can be arbitrary text in between (.*?).
    The tags don’t need to be on separate lines.
    Any amount of whitespace (\s*) is allowed between the </reasoning> and <answer>.
    Minor formatting variations (e.g., extra spaces or missing newlines) will still get a 0.5 reward.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


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
    # CosineScaledSparseReward hyperparameters:
    min_value_wrong: float = -2.0,
    max_value_wrong: float = 0.0,
    min_value_correct: float = 0.0,
    max_value_correct: float = 2.0,
    max_len: int = 200,
    exceed_length: float = -2.0,
    repetition_max_penalty: float = -1.0,
    repetition_ngram_size: int = 3,
    **kwargs
) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")

    scores = [1.0 if er == ans else 0.0 for er, ans in zip(extracted_responses, answer)]
    # TODO: check whether should base this on tokens?
    gen_lengths = [len(r.split()) for r in responses]

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

    rewards = cos_reward.reward(
        sequences=responses,
        gen_lengths=gen_lengths,
        scores=scores
    )

    return rewards


def zipngram_tokens(tokens: list[int], ngram_size: int):
    return zip(*[tokens[i:] for i in range(ngram_size)])


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
