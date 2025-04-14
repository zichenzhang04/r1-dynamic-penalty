import math
import wandb
from dynamic_penalty.train.cosine import CosineScaledSparseReward
from dynamic_penalty.train.utils import average_nonzero


class DynamicWeightedReward:
    """
    Dynamic reward function that adjusts the weight of repetition penalty based on
    the repetition frequency in the generated sequence.

    R_dynamic = (1 - w_repetition) * R_cosine - w_repetition * sum(P_repetition(x_t))

    where w_repetition = sigmoid(alpha * f_rep(X) + beta)
    """

    def __init__(
        self,
        min_value_wrong: float = -10.0,
        max_value_wrong: float = 0.0,
        min_value_correct: float = 2.0,
        max_value_correct: float = 1.0,
        max_len: int = 1024,
        exceed_length: float = -10.0,
        repetition_max_penalty: float = -1.0,
        repetition_ngram_size: int = 20,
        alpha: float = 2.0,  # Controls how quickly weight changes with repetition frequency
        beta: float = -1.0,  # Base bias for sigmoid function
    ):
        self.cosine_reward = CosineScaledSparseReward(
            min_value_wrong,
            max_value_wrong,
            min_value_correct,
            max_value_correct,
            max_len,
            exceed_length,
            repetition_max_penalty,
            repetition_ngram_size,
        )
        self.alpha = alpha
        self.beta = beta
        self.repetition_ngram_size = repetition_ngram_size
        self.repetition_max_penalty = repetition_max_penalty

    def calculate_repetition_frequency(self, text: str) -> float:
        """
        Calculate the repetition frequency in the generated sequence.
        Returns a value between 0 and 1 representing the fraction of repeated n-grams.
        """
        words = text.lower().split()
        if len(words) <= self.repetition_ngram_size:
            return 0.0

        ngrams = set()
        repeated_count = 0
        total_ngrams = len(words) - self.repetition_ngram_size + 1

        for i in range(total_ngrams):
            ngram = tuple(words[i:i + self.repetition_ngram_size])
            if ngram in ngrams:
                repeated_count += 1
            else:
                ngrams.add(ngram)

        return repeated_count / total_ngrams if total_ngrams > 0 else 0.0

    def calculate_dynamic_weight(self, repetition_freq: float) -> float:
        """
        Calculate the dynamic weight using sigmoid function:
        w_repetition = sigmoid(alpha * f_rep(X) + beta)
        """
        return 1.0 / (1.0 + math.exp(-(self.alpha * repetition_freq + self.beta)))

    def reward(self, sequences, gen_lengths, scores):
        """
        Calculate the dynamic weighted reward.

        Args:
            sequences: List of generated text sequences
            gen_lengths: List of lengths of the generated sequences
            scores: List of correctness scores (1.0 for correct, 0.0 for wrong)

        Returns:
            List of reward values
        """
        # Get rewards from cosine reward function (without applying penalties)
        cosine_rewards, _ = self.cosine_reward.reward(
            sequences, gen_lengths, scores
        )

        # Track metrics
        rep_frequencies = []
        rep_penalties = []
        dynamic_rewards = []
        weights = []

        # Process each sequence
        for i, (seq, cosine_reward) in enumerate(zip(sequences, cosine_rewards)):
            # Calculate repetition frequency
            print(sequences)
            rep_freq = self.calculate_repetition_frequency(seq)
            rep_frequencies.append(rep_freq)

            # Calculate dynamic weight based on repetition frequency
            weight = self.calculate_dynamic_weight(rep_freq)
            weights.append(weight)

            # Calculate repetition penalty ourselves
            rep_penalty = 0.0
            words = seq.lower().split()
            ngrams = set()
            for j in range(len(words) - self.repetition_ngram_size + 1):
                ngram = tuple(words[j:j + self.repetition_ngram_size])
                if ngram in ngrams:
                    rep_penalty -= abs(self.repetition_max_penalty)
                else:
                    ngrams.add(ngram)

            rep_penalties.append(rep_penalty)

            # Calculate dynamic reward
            # R_dynamic = (1 - w_repetition) * R_cosine - w_repetition * |rep_penalty|
            dynamic_reward = (1.0 - weight) * cosine_reward - \
                weight * abs(rep_penalty)
            dynamic_rewards.append(dynamic_reward)

        # Log metrics
        if weights:
            wandb.log({"train/dynamic_weight": average_nonzero(weights)})
            wandb.log(
                {"train/repetition_frequency": average_nonzero(rep_frequencies)})
            if rep_penalties:
                wandb.log(
                    {"train/repetition_penalty": average_nonzero([abs(p) for p in rep_penalties if p != 0])})

        return dynamic_rewards
