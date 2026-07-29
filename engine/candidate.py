class CandidateSelector:
    """
    점수가 높은 번호를 후보군으로 선택한다.
    Generator는 번호(int) 리스트를 입력으로 받으므로
    NumberScore -> int 변환까지 담당한다.
    """

    def __init__(self):
        pass

    def select(self, scores, limit=18):

        ordered = sorted(
            scores,
            key=lambda s: s.total_score,
            reverse=True,
        )

        return [
            score.number
            for score in ordered[:limit]
        ]