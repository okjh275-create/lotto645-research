from engine.feature import FeatureEngine
from engine.score import ScoreEngine
from engine.candidate import CandidateSelector
from engine.generator import GeneratorEngine


class Predictor:

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.score_engine = ScoreEngine()
        self.selector = CandidateSelector()
        self.generator = GeneratorEngine()

    def predict(self):

        # 1. Feature 생성
        features = self.feature_engine.build()

        # 2. 번호별 점수 계산
        scores = [
            self.score_engine.build(f)
            for f in features
        ]

        # 3. 상위 후보 선택
        candidates = self.selector.select(
            scores,
            limit=18,
        )

        # 4. 조합 생성
        return self.generator.generate(
            candidates,
            count=5,
        )