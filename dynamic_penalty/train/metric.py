import re

def count_reasoning_words(text):
    """Extract text inside <reasoning>...</reasoning> and count words."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)  # Extract reasoning section
    if match:
        reasoning_text = match.group(1).strip()  # Get content inside <reasoning>
        word_count = len(reasoning_text.split())  # Count words
    else:
        word_count = 0  # No reasoning tag found

    return word_count
