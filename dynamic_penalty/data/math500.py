from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Reasoning process here
</reasoning>
<answer>
An integer
</answer>
Only output the numerical values (integers) of final answers between <answer> tags
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
def get_math500_questions_eval(subset='int_only', limit=256) -> Dataset:
    """
    Params:
    subset: if set to 'all', then all the 500 entries are sampled; 
    if set to 'int_only', then only entries with integer gt are selected (totally 293)
    limit: the maximum number of selected entries
    """
    # Auto-saved to ~/.cache/huggingface/datasets/
    data = load_dataset("HuggingFaceH4/MATH-500")["test"] # type: ignore
    if subset == 'int_only':
        data_ls = []
        for i, entry in enumerate(data):
            if entry['answer'].strip().isdigit():
                data_ls.append(entry)
    
        data = Dataset.from_list(data_ls)       

    if len(data) > limit:
        data = data.select(range(limit))

    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer'].strip()
    }) # type: ignore
    
    return data # type: ignore
