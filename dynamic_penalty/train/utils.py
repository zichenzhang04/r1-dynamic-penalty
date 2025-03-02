def zipngram_tokens(tokens: list[int], ngram_size: int):
    tokens = tokens.lower().split()
    return zip(*[tokens[i:] for i in range(ngram_size)])
