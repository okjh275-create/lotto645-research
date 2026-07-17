from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def predict(self, features):
        raise NotImplementedError


class FrequencyStrategy(Strategy):

    def predict(self, features):

        ordered = sorted(
            features,
            key=lambda f: (
                f.freq10,
                f.freq20,
                f.freq50,
                f.freq_all,
            ),
            reverse=True,
        )

        return sorted(
            [f.number for f in ordered[:6]]
        )