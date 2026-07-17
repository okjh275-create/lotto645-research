from engine.replay import ReplayEngine


def main():

    predicted = [4, 13, 18, 41, 42, 44]
    actual = [7, 18, 28, 33, 39, 44]

    hit = ReplayEngine.count_hits(
        predicted,
        actual,
    )

    print("=" * 50)
    print("Replay Test")
    print("=" * 50)
    print("Predicted :", predicted)
    print("Actual    :", actual)
    print("Hits      :", hit)
    print("=" * 50)


if __name__ == "__main__":
    main()