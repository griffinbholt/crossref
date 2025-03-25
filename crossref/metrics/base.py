import numpy as np

from abc import ABC, abstractmethod


class SimilarityMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, text1: str, text2: str, **kwargs) -> float:
        pass

    def score_all(self, texts1: list[str], texts2: list[str], **kwargs) -> np.ndarray:
        """Default `score_all` implementation using `score`."""
        return np.array([[self.score(text1, text2, **kwargs) for text2 in texts2] for text1 in texts1])
