from dataclasses import dataclass

from engine.predictor import Predictor
from engine.statistics import StatisticsEngine
from engine.replay import ReplayEngine
from engine.metrics import MetricsEngine


@dataclass
class BacktestResult:
    rounds: list
    metrics: MetricsEngine


class BacktestEngine:

    def __init__(self):
        self.stats = StatisticsEngine()
        self.predictor = Predictor()

    def run(self, start_round, end_round):

        hits = []

        for target_round in range(start_round, end_round + 1):

            predicted_sets = self.predictor.predict(
                until_round=target_round - 1
            )

            actual = self.stats.get_draw(target_round)

            if actual is None:
                continue

            best_hit = 0

            for numbers in predicted_sets:

                hit = ReplayEngine.count_hits(
                    numbers,
                    actual,
                )

                if hit > best_hit:
                    best_hit = hit

            hits.append(best_hit)

        return MetricsEngine.summarize(hits)