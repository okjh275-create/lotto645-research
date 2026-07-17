from engine.filter import FilterEngine


def main():
    engine = FilterEngine()

    tests = [
        [1, 2, 3, 4, 5, 6],
        [7, 9, 13, 22, 31, 44],
        [4, 13, 18, 31, 42, 44],
    ]

    print("=" * 50)
    print("Filter Test")
    print("=" * 50)

    for numbers in tests:
        print(numbers, "->", engine.is_valid(numbers))


if __name__ == "__main__":
    main()