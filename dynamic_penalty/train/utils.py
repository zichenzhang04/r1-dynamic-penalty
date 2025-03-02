def zipngram_tokens(tokens: list[int], ngram_size: int):
    tokens = tokens.lower().split()
    return zip(*[tokens[i:] for i in range(ngram_size)])

def average_nonzero(lst):
    nonzero_elements = [x for x in lst if x != 0]
    return sum(nonzero_elements) / len(nonzero_elements) if nonzero_elements else 0  # Avoid division by zero
