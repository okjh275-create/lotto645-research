import random

from engine.filter import FilterEngine


class GeneratorEngine:

    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.filter = FilterEngine()

    def generate(self, candidates, count=5):

        results = []

        if len(candidates) < 6:
            return results

        while len(results) < count:

            selected = self.random.choices(
                population=candidates,
                weights=[c.total_score for c in candidates],
                k=12,
            )

            unique = []

            seen = set()

            for item in selected:

                if item.number not in seen:
                    unique.append(item.number)
                    seen.add(item.number)

                if len(unique) == 6:
                    break

            if len(unique) < 6:
                continue

            numbers = sorted(unique)

            if self.filter.is_valid(numbers):
                if numbers not in results:
                    results.append(numbers)

        return results