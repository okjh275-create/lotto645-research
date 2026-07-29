from engine.metrics import MetricsEngine


class BenchmarkEngine:

    @staticmethod
    def evaluate(replay_results):
        """
        replay_results : ReplayResult 리스트
        """

        hits = [r.hit_count for r in replay_results]

        return MetricsEngine.summarize(hits)