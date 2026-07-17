from engine.metrics import MetricsEngine


def main():

    hits = [2, 1, 3, 2, 4, 1, 0, 5, 2, 3]

    result = MetricsEngine.summarize(hits)

    print("=" * 60)
    print("Benchmark")
    print("=" * 60)
    print("Total Replay :", result.total)
    print(f"Average Hit  : {result.average_hit:.2f}")
    print("3+ Hits      :", result.hit3)
    print("4+ Hits      :", result.hit4)
    print("5+ Hits      :", result.hit5)
    print("6 Hits       :", result.hit6)
    print("=" * 60)


if __name__ == "__main__":
    main()