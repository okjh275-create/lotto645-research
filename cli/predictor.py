from engine.predictor import Predictor


def main():

    predictor = Predictor()

    results = predictor.predict()

    print("=" * 50)
    print("Lotto645 Predictor")
    print("=" * 50)

    for i, numbers in enumerate(results, start=1):
        print(f"SET {i}: {numbers}")

    print("=" * 50)


if __name__ == "__main__":
    main()