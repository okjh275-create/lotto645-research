from engine.statistics import StatisticsEngine


def main():
    stats = StatisticsEngine()

    print("=" * 60)
    print("Lotto645 Statistics")
    print("=" * 60)

    print("\n[Top 10 Frequency - All Draws]")
    for num, cnt in stats.frequency().most_common(10):
        print(f"{num:2d} : {cnt}")

    print("\n[Top 10 Frequency - Recent 10]")
    for num, cnt in stats.frequency(10).most_common(10):
        print(f"{num:2d} : {cnt}")

    print("\n[Top 10 Frequency - Recent 20]")
    for num, cnt in stats.frequency(20).most_common(10):
        print(f"{num:2d} : {cnt}")

    print("\n[Top 10 Frequency - Recent 50]")
    for num, cnt in stats.frequency(50).most_common(10):
        print(f"{num:2d} : {cnt}")

    print("\n[Longest Gap Top 10]")

    gaps = stats.gap()

    for number, gap in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{number:2d} : {gap}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()