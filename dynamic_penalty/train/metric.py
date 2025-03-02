import re
from dynamic_penalty.data.gsm8k import extract_xml_answer

def count_reasoning_words(text):
    """Extract text inside <reasoning>...</reasoning> and count words."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)  # Extract reasoning section
    if match:
        reasoning_text = match.group(1).strip()  # Get content inside <reasoning>
        word_count = len(reasoning_text.split())  # Count words
    else:
        word_count = 0  # No reasoning tag found

    return word_count

# def training_accuracy(responses, answer):
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     res = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
#     return sum(res) / len(res)
