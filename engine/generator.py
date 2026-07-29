import random

from engine.filter import FilterEngine
from engine.diversity import DiversityEngine
from engine.config import Config


class GeneratorEngine:

    def __init__(self, seed=None):

        if seed is None:
            seed = Config.random_seed()

        self.random = random.Random(seed)
        self.filter = FilterEngine()

    def generate(
        self,
        candidates,
        count=None,
    ):

        if count is None:
            count = Config.predict_count()

        if len(candidates) < 6:
            return []

        sample_count = Config.generator_samples()

        pool = []

        for _ in range(sample_count):

            numbers = sorted(
                self.random.sample(
                    candidates,
                    6,
                )
            )

            if not self.filter.is_valid(numbers):
                continue

            if numbers in pool:
                continue

            pool.append(numbers)

        results = []

        for numbers in pool:

            duplicated = False

            for selected in results:

                if (
                    DiversityEngine.overlap(
                        numbers,
                        selected,
                    )
                    >= 4
                ):
                    duplicated = True
                    break

            if duplicated:
                continue

            results.append(numbers)

            if len(results) >= count:
                break

        return results