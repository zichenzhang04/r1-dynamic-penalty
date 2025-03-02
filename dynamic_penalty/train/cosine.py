import math


class CosineScaledSparseReward:
    """
    Adapted from
    https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/cosine.py
    """
    def __init__(
        self,
        min_value_wrong: float,
        max_value_wrong: float,
        min_value_correct: float,
        max_value_correct: float,
        max_len: int,
        exceed_length: float,
        repetition_max_penalty: float,
        repetition_ngram_size: int,
    ):
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct
        self.max_len = max_len
        self.exceed_length = exceed_length
        self.repetition_max_penalty = repetition_max_penalty
        self.repetition_ngram_size = repetition_ngram_size
        self.MAX_LEN_MARGIN = 16 # TODO: adjust this

    def get_repetition_penalty(self, text: str) -> float:
        words = text.lower().split()
        ngrams = set()
        penalty = 0.0
        for i in range(len(words) - self.repetition_ngram_size + 1):
            ngram = tuple(words[i : i + self.repetition_ngram_size])
            if ngram in ngrams:
                penalty -= abs(self.repetition_max_penalty)
            else:
                ngrams.add(ngram)
        return penalty

    def reward(
        self,
        sequences,
        gen_lengths,
        scores
    ):
        """Calculate correct/wrong rewards based solution length using a cosine schedule.

        The general idea is:
        - Shorter correct solutions should be rewarded over longer ones.
        - Longer wrong solutions should be rewarded over shorter ones.
        - Shorter solutions should be more risk averse (wrong penalized more than correct rewarded).
        """
        rewards = []
        rep_penalties = []

        for seq, length, score in zip(sequences, gen_lengths, scores):
            if length + self.MAX_LEN_MARGIN >= self.max_len:
                rewards.append(self.exceed_length)
                # rep_penalties.append(0)
                continue

            if score >= 1.0: # Correct
                min_value = self.max_value_correct # 1
                max_value = self.min_value_correct # 2
                rep_penalty = 0
            else: # Wrong
                min_value = self.max_value_wrong # 0
                max_value = self.min_value_wrong # -10
                rep_penalty = self.get_repetition_penalty(seq)
                if rep_penalty != 0:
                    rep_penalties.append(rep_penalty)

            progress = length / self.max_len
            assert 0 <= progress <= 1, "Cosine Reward Error: length / self.max_len should be in [0, 1]"
            r = compute_cosine(min_value, max_value, progress)
            r += rep_penalty
            rewards.append(r)

        return rewards, rep_penalties


def compute_cosine(min_value, max_value, progress):
    return min_value + 0.5 * (max_value - min_value) * (1.0 + math.cos(progress * math.pi))
