from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path

from engine.backtest import BacktestEngine


@dataclass
class ExperimentResult:
    name: str
    average_hit: float
    hit3: int
    hit4: int
    hit5: int
    hit6: int


class ExperimentEngine:

    def __init__(self):
        self.backtest = BacktestEngine()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def run(self, name, start_round, end_round):

        result = self.backtest.run(
            start_round,
            end_round,
        )

        logfile = self.log_dir / "experiments.csv"

        new_file = not logfile.exists()

        with logfile.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            if new_file:
                writer.writerow([
                    "datetime",
                    "name",
                    "average_hit",
                    "hit3",
                    "hit4",
                    "hit5",
                    "hit6",
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                round(result.average_hit, 3),
                result.hit3,
                result.hit4,
                result.hit5,
                result.hit6,
            ])

        return ExperimentResult(
            name=name,
            average_hit=result.average_hit,
            hit3=result.hit3,
            hit4=result.hit4,
            hit5=result.hit5,
            hit6=result.hit6,
        )