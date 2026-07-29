from engine.feature import FeatureEngine
from engine.score import ScoreEngine
from engine.ranking import RankingEngine


def main():

    feature_engine = FeatureEngine()
    score_engine = ScoreEngine()

    features = feature_engine.build()

    scores = [
        score_engine.build(f)
        for f in features
    ]

    ranked = RankingEngine.rank(scores)

    print("=" * 60)
    print("TOP 20 RANKING")
    print("=" * 60)

    for r in ranked[:20]:
        print(
            f"{r.number:2d}",
            f"{r.score:.2f}",
        )


if __name__ == "__main__":
    main()