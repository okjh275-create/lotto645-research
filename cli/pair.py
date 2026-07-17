from engine.pair import PairEngine


def main():

    engine = PairEngine()

    print("=" * 60)
    print("Lotto645 Pair Statistics")
    print("=" * 60)

    print("\n[Top 20 Pair Frequency - All Draws]")

    for pair, count in engine.pair_frequency().most_common(20):
        print(f"{pair[0]:2d} {pair[1]:2d} : {count}")

    print("\n[Top 20 Pair Frequency - Recent 50 Draws]")

    for pair, count in engine.pair_frequency(50).most_common(20):
        print(f"{pair[0]:2d} {pair[1]:2d} : {count}")

    print("\n[Top 20 Pair Frequency - Recent 20 Draws]")

    for pair, count in engine.pair_frequency(20).most_common(20):
        print(f"{pair[0]:2d} {pair[1]:2d} : {count}")

    print("\n[Top 20 Pair Frequency - Recent 10 Draws]")

    for pair, count in engine.pair_frequency(10).most_common(20):
        print(f"{pair[0]:2d} {pair[1]:2d} : {count}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()