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

    @classmethod
    def replay(
        cls,
        train_last_round,
        test_round,
        predicted,
        actual,
    ):
        return ReplayResult(
            train_last_round=train_last_round,
            test_round=test_round,
            predicted=predicted,
            actual=actual,
            hit_count=cls.count_hits(predicted, actual),
        )