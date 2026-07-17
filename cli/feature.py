from engine.feature import FeatureEngine


def main():

    engine = FeatureEngine()

    features = engine.build()

    print("=" * 70)
    print("Feature Engine")
    print("=" * 70)

    print(
        f"{'No':>2} {'All':>4} {'10':>4} {'20':>4} {'50':>4} {'Gap':>4}"
    )

    for f in features:
        print(
            f"{f.number:2d} "
            f"{f.freq_all:4d} "
            f"{f.freq10:4d} "
            f"{f.freq20:4d} "
            f"{f.freq50:4d} "
            f"{f.gap:4d}"
        )


if __name__ == "__main__":
    main()