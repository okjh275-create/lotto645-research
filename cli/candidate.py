from engine.candidate import CandidateSelector
from engine.score import NumberScore


def main():
    selector = CandidateSelector()

    scores = [
        NumberScore(1, 0.8, 0.2, 0.0, 1.0),
        NumberScore(2, 0.3, 0.1, 0.0, 0.4),
        NumberScore(3, 0.9, 0.4, 0.0, 1.3),
        NumberScore(4, 0.5, 0.3, 0.0, 0.8),
        NumberScore(5, 0.6, 0.1, 0.0, 0.7),
    ]

    result = selector.select(scores, limit=3)

    print("=" * 50)
    print("Candidate Selector Test")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()