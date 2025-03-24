from abc import ABC, abstractmethod


class ComparisonScorer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, text1: str, text2: str, **kwargs) -> float:
        pass
