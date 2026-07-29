from dataclasses import dataclass

from engine.statistics import StatisticsEngine


@dataclass
class NumberFeature:
    number: int
    freq_all: int
    freq10: int
    freq20: int
    freq50: int
    gap: int
    score: float = 0.0


class FeatureEngine:

    def __init__(self):
        self.stats = StatisticsEngine()

    def build(self, until_round=None):

        freq_all = self.stats.frequency(
            until_round=until_round,
        )

        freq10 = self.stats.frequency(
            10,
            until_round=until_round,
        )

        freq20 = self.stats.frequency(
            20,
            until_round=until_round,
        )

        freq50 = self.stats.frequency(
            50,
            until_round=until_round,
        )

        gaps = self.stats.gap(
            until_round=until_round,
        )

        features = []

        for number in range(1, 46):

            features.append(
                NumberFeature(
                    number=number,
                    freq_all=freq_all.get(number, 0),
                    freq10=freq10.get(number, 0),
                    freq20=freq20.get(number, 0),
                    freq50=freq50.get(number, 0),
                    gap=gaps[number],
                )
            )

        return features