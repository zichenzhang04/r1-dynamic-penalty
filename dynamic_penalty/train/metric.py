import re
from dynamic_penalty.data.gsm8k import extract_xml_answer
import wandb


def count_reasoning_words(text):
    """Extract text inside <reasoning>...</reasoning> and count words."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)  # Extract reasoning section
    if match:
        reasoning_text = match.group(1).strip()  # Get content inside <reasoning>
        word_count = len(reasoning_text.split())  # Count words
    else:
        word_count = 0  # No reasoning tag found

    return word_count


def count_aha_words(responses, aha_words=['wait', 'recheck', 'alternatively', 'retry', 'however']):
    """Count the frequency of each aha words. responses should be a list containing multiple texts"""
    num_texts = len(responses)
    cnt_dict = dict([(aha_word, 0) for aha_word in aha_words])
    cnt_dict['cnt_all_aha_words'] = 0

    for text in responses:
        for aha_word in aha_words:
            num_aha_word = len(re.findall(rf'\b{aha_word}\b', text, flags=re.IGNORECASE))
            cnt_dict[aha_word] += num_aha_word
            cnt_dict['cnt_all_aha_words'] += num_aha_word
    for key in cnt_dict.keys():
        cnt_dict[key] /= num_texts

    return cnt_dict


def log_aha_words(responses):
    # Calculate the average number of each aha word across the responses
    aha_words = ['wait', 'recheck', 'alternatively', 'retry', 'however']
    cnt_dict = count_aha_words(responses, aha_words=aha_words)
    for aha_word in aha_words:
        wandb.log({f"train/cnt_'{aha_word}'": cnt_dict[aha_word]})
    wandb.log({"train/cnt_all_aha_words": cnt_dict['cnt_all_aha_words']})

# def training_accuracy(responses, answer):
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     res = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
#     return sum(res) / len(res)
