from engine.generator import GeneratorEngine


def main():
    engine = GeneratorEngine(seed=20260717)

    candidates = [
        4, 7, 9, 13, 15,
        18, 22, 28, 31,
        38, 41, 42, 44, 45
    ]

    results = engine.generate(
        candidates,
        count=5
    )

    print("=" * 50)
    print("Generator Test")
    print("=" * 50)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()