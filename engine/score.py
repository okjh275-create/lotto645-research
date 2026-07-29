from dataclasses import dataclass

from engine.config import Config
from engine.pair import PairEngine


@dataclass
class NumberScore:
    number: int
    freq_score: float
    gap_score: float
    pair_score: float
    total_score: float


class ScoreEngine:

    def __init__(self):
        self.weights = Config.weights()
        self.pair_engine = PairEngine()

    def build(self, feature, until_round=None):

        freq = (
            feature.freq10 * self.weights["freq10"]
            + feature.freq20 * self.weights["freq20"]
            + feature.freq50 * self.weights["freq50"]
        )

        gap = feature.gap * self.weights["gap"]

        pair_scores = self.pair_engine.number_scores(
            last_n=50,
            until_round=until_round,
        )

        pair = pair_scores.get(
            feature.number,
            0,
        )

        return NumberScore(
            number=feature.number,
            freq_score=freq,
            gap_score=gap,
            pair_score=pair,
            total_score=freq + gap + pair,
        )