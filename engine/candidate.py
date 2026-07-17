class CandidateSelector:

    def __init__(self):
        pass

    def select(self, scores, limit=18):
        """
        scores : NumberScore 리스트
        limit  : 선택할 후보 개수
        """

        ordered = sorted(
            scores,
            key=lambda x: x.total_score,
            reverse=True,
        )

        return ordered[:limit]