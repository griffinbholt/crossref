import logging

import numpy as np

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _score_row(args: tuple) -> list:
    """ProcessPoolExecutor row worker: applies fn(t1, t2) for each t2 in corpus.

    Module-level so it is picklable. fn must be a picklable callable (e.g. a
    bound method of a picklable metric object).
    """
    fn, t1, corpus = args
    return [fn(t1, t2) for t2 in corpus]


def _score_self_row(args: tuple) -> dict:
    """ProcessPoolExecutor upper-triangle row worker.

    Returns {j: fn(ti, tj)} for all j > i. Module-level for picklability.
    """
    fn, i, ti, corpus = args
    return {j: fn(ti, corpus[j]) for j in range(i + 1, len(corpus))}


class SimilarityMetric(ABC):
    name: str = ""  # subclasses must override with a unique, non-empty string

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce that classes providing a concrete score() also define a name.
        # We check cls.__dict__ rather than __abstractmethods__ because ABCMeta
        # sets __abstractmethods__ *after* __init_subclass__ is called, making
        # it unreliable here. Abstract intermediates (e.g. SyntacticSimilarityMetric)
        # simply pass without defining score(), so they are naturally exempt.
        if 'score' in cls.__dict__ and not cls.name:
            raise TypeError(
                f"Concrete metric class {cls.__name__!r} must define a non-empty "
                f"class attribute 'name'. Example: name = 'my_metric'"
            )

    @abstractmethod
    def score(self, text1: str, text2: str, **kwargs) -> float:
        pass

    def score_all(self, texts1: list[str], texts2: list[str], **kwargs) -> np.ndarray:
        """Default score_all implementation using score(). Override for efficiency."""
        logger.debug("%s.score_all: %dx%d", self.__class__.__name__, len(texts1), len(texts2))
        return np.array([[self.score(t1, t2, **kwargs) for t2 in texts2] for t1 in texts1])

    def score_self(self, texts: list[str], **kwargs) -> np.ndarray:
        """Score a list of texts against itself (self-comparison).

        Override in subclasses to exploit self-compare structure — e.g. computing
        only the upper triangle and mirroring, or training a single model on the
        full corpus rather than fitting twice.

        The returned matrix must be symmetric with ones on the diagonal for
        metrics where score(text, text) == 1.0.
        """
        return self.score_all(texts, texts, **kwargs)

    def config(self) -> dict:
        """Return a serializable dict sufficient to reconstruct this metric.

        Override in subclasses to include metric-specific parameters.
        The returned dict must include a 'name' key matching the class name attribute.
        """
        return {"name": self.name}

    def explain(self, text1: str, text2: str) -> dict:
        """Return evidence for why text1 and text2 are similar.

        Override in subclasses to provide metric-specific evidence.
        Default implementation returns just the score.
        """
        return {"score": self.score(text1, text2)}
