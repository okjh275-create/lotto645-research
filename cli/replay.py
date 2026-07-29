from engine.replay import ReplayEngine


def main():

    result = ReplayEngine.replay(
        train_last_round=1230,
        test_round=1231,
        predicted=[4, 13, 18, 41, 42, 44],
        actual=[7, 18, 28, 33, 39, 44],
    )

    print("=" * 50)
    print("Replay Test")
    print("=" * 50)
    print("Train Last :", result.train_last_round)
    print("Test Round :", result.test_round)
    print("Predicted  :", result.predicted)
    print("Actual     :", result.actual)
    print("Hits       :", result.hit_count)
    print("=" * 50)


if __name__ == "__main__":
    main()