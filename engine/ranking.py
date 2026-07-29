from dataclasses import dataclass


@dataclass
class RankedNumber:
    number: int
    score: float


class RankingEngine:

    @staticmethod
    def rank(scores):

        ordered = sorted(
            scores,
            key=lambda s: s.total_score,
            reverse=True,
        )

        return [
            RankedNumber(
                number=s.number,
                score=s.total_score,
            )
            for s in ordered
        ]