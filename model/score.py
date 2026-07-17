from engine.config import Config


class ScoreEngine:

    def __init__(self):
        self.weights = Config.weights()

    def build(self, feature):
        freq = (
            feature["freq10"] * self.weights["freq10"]
            + feature["freq20"] * self.weights["freq20"]
            + feature["freq50"] * self.weights["freq50"]
        )

        gap = feature["gap"] * self.weights["gap"]

        total = freq + gap

        return NumberScore(
            number=feature["number"],
            freq_score=freq,
            gap_score=gap,
            pair_score=0.0,
            total_score=total,
        )