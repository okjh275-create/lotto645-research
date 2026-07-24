"""Strategy-performance statistics models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyStatistics:
    """Incrementally accumulated performance for one strategy."""

    strategy_type: str
    strategy_name: str
    sample_count: int
    total_matches: int
    total_prediction_score: float
    hit3_count: int
    hit4_count: int
    hit5_count: int
    hit6_count: int
    prize_count: int
    updated_at_kst: str

    @property
    def average_match_count(self) -> float:
        if self.sample_count == 0:
            return 0.0

        return self.total_matches / self.sample_count

    @property
    def average_prediction_score(self) -> float:
        if self.sample_count == 0:
            return 0.0

        return self.total_prediction_score / self.sample_count

    @property
    def hit3_plus_rate(self) -> float:
        if self.sample_count == 0:
            return 0.0

        hit_count = (
            self.hit3_count
            + self.hit4_count
            + self.hit5_count
            + self.hit6_count
        )
        return hit_count / self.sample_count

    @property
    def prize_rate(self) -> float:
        if self.sample_count == 0:
            return 0.0

        return self.prize_count / self.sample_count

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "sample_count": self.sample_count,
            "average_match_count": round(
                self.average_match_count,
                6,
            ),
            "average_prediction_score": round(
                self.average_prediction_score,
                6,
            ),
            "hit3_count": self.hit3_count,
            "hit4_count": self.hit4_count,
            "hit5_count": self.hit5_count,
            "hit6_count": self.hit6_count,
            "hit3_plus_rate": round(
                self.hit3_plus_rate,
                6,
            ),
            "prize_count": self.prize_count,
            "prize_rate": round(
                self.prize_rate,
                6,
            ),
            "updated_at_kst": self.updated_at_kst,
        }
