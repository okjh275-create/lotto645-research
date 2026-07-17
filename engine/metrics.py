from dataclasses import dataclass


@dataclass
class MetricsResult:
    total: int
    average_hit: float
    hit3: int
    hit4: int
    hit5: int
    hit6: int


class MetricsEngine:

    @staticmethod
    def summarize(hit_list):

        total = len(hit_list)

        if total == 0:
            return MetricsResult(0, 0, 0, 0, 0, 0)

        return MetricsResult(
            total=total,
            average_hit=sum(hit_list) / total,
            hit3=sum(x >= 3 for x in hit_list),
            hit4=sum(x >= 4 for x in hit_list),
            hit5=sum(x >= 5 for x in hit_list),
            hit6=sum(x == 6 for x in hit_list),
        )