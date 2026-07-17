from engine.config import Config


class FilterEngine:
    def __init__(self):
        self.filters = Config.filters()

    def is_valid(self, numbers):
        """
        numbers : 정렬된 6개 번호 리스트
        """

        if len(numbers) != 6:
            return False

        # -------------------------
        # 합계 검사
        # -------------------------
        total = sum(numbers)

        if not (
            self.filters["sum_min"]
            <= total
            <= self.filters["sum_max"]
        ):
            return False

        # -------------------------
        # 홀짝 검사
        # -------------------------
        odd = sum(n % 2 for n in numbers)
        even = 6 - odd

        if odd < self.filters["odd_min"]:
            return False

        if odd > self.filters["odd_max"]:
            return False

        # -------------------------
        # 통과
        # -------------------------
        return True