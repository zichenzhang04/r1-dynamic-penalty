from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Reasoning process here
</reasoning>
<answer>
Your answer here
</answer>
Only output your final answer between <answer> tags
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# uncomment middle messages for 1-shot prompting
def get_theoremqa_questions(split="test", limit=None) -> Dataset:
    """
    Params:
    split: dataset split to use (only 'test' is available for this dataset)
    limit: the maximum number of selected entries (if None, use all available data)
    """
    # Auto-saved to ~/.cache/huggingface/datasets/
    data = load_dataset("TIGER-Lab/TheoremQA")[split] # type: ignore
    
    if limit is not None and len(data) > limit:
        data = data.select(range(limit))

    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['Question']}
        ],
        'answer': x['Answer'],
        'answer_type': x['Answer_type']
    }) # type: ignore
    
    return data # type: ignore 