from types import MethodType
from symeval import EvaluatorMathBatch
from dynamic_penalty.train.sub_functions import _prepare_inputs

def zipngram_tokens(tokens: list[int], ngram_size: int):
    tokens = tokens.lower().split()
    return zip(*[tokens[i:] for i in range(ngram_size)])


def average_nonzero(lst):
    nonzero_elements = [x for x in lst if x != 0]
    return sum(nonzero_elements) / len(nonzero_elements) if nonzero_elements else 0  # Avoid division by zero


def customize_trainer(trainer):
    # substitude the _prepare_inputs function in trainer, to enable correct eval logging
    trainer._prepare_inputs = MethodType(_prepare_inputs, trainer)


def math_equal(model_answer, true_answer):
    """
    Compares two mathematical expressions for equivalence.
    """
    # print(f"[Model answer]: {model_answer}")
    # print(f"[True answer]: {true_answer}")
    evaluator = EvaluatorMathBatch()
    if model_answer is None or true_answer is None:
        return False
    try:
        return evaluator.eq(model_answer, true_answer)
    except Exception as e:
        print(e)
        return model_answer == true_answer
