from engine.statistics import StatisticsEngine


def main():
    stats = StatisticsEngine()

    print("=" * 50)
    print("Statistics Report")
    print("=" * 50)

    # 전체 빈도
    freq_all = stats.frequency()

    print("\nTop 10 Frequency (All)")
    for num, cnt in freq_all.most_common(10):
        print(f"{num:2d} : {cnt}")

    # 최근 10회
    print("\nTop Frequency (Recent 10)")
    for num, cnt in stats.frequency(10).most_common(10):
        print(f"{num:2d} : {cnt}")

    # 최근 20회
    print("\nTop Frequency (Recent 20)")
    for num, cnt in stats.frequency(20).most_common(10):
        print(f"{num:2d} : {cnt}")

    # 최근 50회
    print("\nTop Frequency (Recent 50)")
    for num, cnt in stats.frequency(50).most_common(10):
        print(f"{num:2d} : {cnt}")

    # Gap
    gaps = stats.gap()

    print("\nLongest Gap Top 10")
    for num, gap in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{num:2d} : {gap}")

    print("=" * 50)


if __name__ == "__main__":
    main()