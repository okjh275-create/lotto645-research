from engine.experiment import ExperimentEngine


def main():

    engine = ExperimentEngine()

    result = engine.run(
        name="baseline_v1",
        start_round=1200,
        end_round=1231,
    )

    print("=" * 60)
    print("EXPERIMENT")
    print("=" * 60)
    print("Name     :", result.name)
    print("Average  :", round(result.average_hit, 3))
    print("3 Hits   :", result.hit3)
    print("4 Hits   :", result.hit4)
    print("5 Hits   :", result.hit5)
    print("6 Hits   :", result.hit6)
    print("=" * 60)


if __name__ == "__main__":
    main()