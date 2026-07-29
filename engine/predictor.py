from engine.feature import FeatureEngine
from engine.score import ScoreEngine
from engine.candidate import CandidateSelector
from engine.generator import GeneratorEngine
from engine.config import Config


class Predictor:

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.score_engine = ScoreEngine()
        self.selector = CandidateSelector()
        self.generator = GeneratorEngine()

    def predict(self, until_round=None):

        # 1. Feature 생성
        features = self.feature_engine.build(
            until_round=until_round,
        )

        # 2. 번호 점수 계산
        scores = [
            self.score_engine.build(
                f,
                until_round=until_round,
            )
            for f in features
        ]

        # 3. 후보번호 선택
        candidates = self.selector.select(
            scores,
            limit=Config.candidate_size(),
        )

        # 4. 조합 생성
        return self.generator.generate(
            candidates,
            count=Config.predict_count(),
        )