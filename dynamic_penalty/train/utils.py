from types import MethodType

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


