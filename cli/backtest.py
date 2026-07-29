from engine.backtest import BacktestEngine


def main():

    engine = BacktestEngine()

    result = engine.run(
        start_round=1200,
        end_round=1231,
    )

    print("=" * 60)
    print("BACKTEST")
    print("=" * 60)

    print("Replay :", result.total)
    print("Average:", round(result.average_hit, 2))
    print("3 Hits :", result.hit3)
    print("4 Hits :", result.hit4)
    print("5 Hits :", result.hit5)
    print("6 Hits :", result.hit6)

    print("=" * 60)


if __name__ == "__main__":
    main()