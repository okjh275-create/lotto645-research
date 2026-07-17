from dataclasses import dataclass


@dataclass
class ReplayResult:
    train_last_round: int
    test_round: int
    predicted: list[int]
    actual: list[int]
    hit_count: int


class ReplayEngine:

    @staticmethod
    def count_hits(predicted, actual):
        return len(set(predicted) & set(actual))