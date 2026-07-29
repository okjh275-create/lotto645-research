import json
from pathlib import Path


class Config:
    ROOT = Path(__file__).resolve().parent.parent
    CONFIG_DIR = ROOT / "config"

    @classmethod
    def _load(cls, filename):
        path = cls.CONFIG_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def settings(cls):
        return cls._load("settings.json")

    @classmethod
    def weights(cls):
        return cls._load("weights.json")

    @classmethod
    def filters(cls):
        return cls._load("filters.json")

    @classmethod
    def candidate_size(cls):
        return cls.settings().get("candidate_size", 18)

    @classmethod
    def generator_samples(cls):
        return cls.settings().get("generator_samples", 10000)

    @classmethod
    def predict_count(cls):
        return cls.settings().get("predict_count", 5)

    @classmethod
    def random_seed(cls):
        return cls.settings().get("random_seed", None)