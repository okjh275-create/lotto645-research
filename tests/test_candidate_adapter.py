from __future__ import annotations

from types import SimpleNamespace
import unittest

from lrp.adapters.candidate import CandidateAdapter
from lrp.contracts import ContractError
from lrp.prediction.probability import (
    NumberProbability,
    ProbabilityVector,
)


def _probability_vector() -> ProbabilityVector:
    probability = 1.0 / 45.0

    return ProbabilityVector(
        round_no=1220,
        generated_at_kst="2026-07-31 18:50",
        probabilities=tuple(
            NumberProbability(
                number=number,
                probability=probability,
                raw_score=1.0,
                rank=number,
                components={
                    "hot": 0.5,
                    "cold": 0.5,
                    "gap": 0.5,
                    "trend": 0.5,
                    "transition": 0.5,
                    "learning": 0.5,
                    "adaptive": 0.5,
                },
                metadata={},
            )
            for number in range(1, 46)
        ),
        metadata={
            "engine": "F-002",
        },
    )


class CandidateAdapterTests(unittest.TestCase):
    def test_probability_mapping_converts_vector(self):
        mapping = CandidateAdapter.probability_mapping(
            _probability_vector()
        )

        self.assertEqual(set(mapping), set(range(1, 46)))
        self.assertAlmostEqual(sum(mapping.values()), 1.0)
        self.assertAlmostEqual(mapping[1], 1.0 / 45.0)
        self.assertAlmostEqual(mapping[45], 1.0 / 45.0)

    def test_probability_mapping_rejects_wrong_type(self):
        with self.assertRaises(ContractError):
            CandidateAdapter.probability_mapping(
                {"number": 1}  # type: ignore[arg-type]
            )

    def test_generate_candidates_forwards_probability_mapping(self):
        captured: dict[str, object] = {}

        def generate_candidates(
            probabilities: object,
            **kwargs: object,
        ) -> str:
            captured["probabilities"] = probabilities
            captured["kwargs"] = kwargs
            return "generated"

        adapter = CandidateAdapter(
            module=SimpleNamespace(
                generate_candidates=generate_candidates,
            )
        )

        probabilities = adapter.probability_mapping(
            _probability_vector()
        )
        result = adapter.generate_candidates(
            probabilities,
            candidate_config="config",
        )

        self.assertEqual(result, "generated")
        self.assertEqual(
            captured["probabilities"],
            probabilities,
        )
        self.assertEqual(
            captured["kwargs"],
            {"candidate_config": "config"},
        )


if __name__ == "__main__":
    unittest.main()
