class DiversityEngine:

    @staticmethod
    def overlap(a, b):
        return len(set(a) & set(b))

    @staticmethod
    def is_diverse(candidate, results, max_overlap=2):
        """
        기존 조합들과 비교하여
        공통 번호가 max_overlap 이하이면 통과
        """

        for numbers in results:
            if DiversityEngine.overlap(candidate, numbers) > max_overlap:
                return False

        return True