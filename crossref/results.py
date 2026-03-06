import pickle

import numpy as np

from typing import Generator, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .documents import Passage
    from .metrics.base import SimilarityMetric


class Match(NamedTuple):
    """A single result from top_k() — one passage from the right side of a comparison.

    Attributes:
        score:  Similarity score.
        index:  Index of this passage in passages2 (pass to explain() as passage2).
        label:  Structural label (e.g. "Book > Chapter > Verse"), or None if unset.
        text:   Passage text.
    """
    score: float
    index: int
    label: str | None
    text: str


class FilterMatch(NamedTuple):
    """A single result from filter() — a passage pair above a similarity threshold.

    Attributes:
        i:       Index of the passage in passages1 (left side).
        j:       Index of the passage in passages2 (right side).
        score:   Similarity score.
        label_i: Structural label of the left passage, or None if unset.
        label_j: Structural label of the right passage, or None if unset.
    """
    i: int
    j: int
    score: float
    label_i: str | None
    label_j: str | None


class Results:
    """Similarity comparison results between two sets of passages.

    Stores one similarity matrix per metric, plus the passage metadata
    for both sides of the comparison.
    """

    def __init__(
        self,
        passages1: list['Passage'],
        passages2: list['Passage'],
        matrices: dict[str, np.ndarray],
        doc1_title: str = "",
        doc2_title: str = "",
        metrics: dict[str, 'SimilarityMetric'] | None = None,
    ):
        self.passages1 = passages1
        self.passages2 = passages2
        self.matrices = matrices
        self.doc1_title = doc1_title
        self.doc2_title = doc2_title
        self._metrics: dict[str, 'SimilarityMetric'] = metrics or {}
        self._metric_configs: dict[str, dict] = {
            name: m.config() for name, m in self._metrics.items()
        }

    def __getstate__(self) -> dict:
        # Exclude live metric objects from pickle — store only their lightweight configs.
        # Metrics will be reconstructed on demand when explain() is called after loading.
        state = self.__dict__.copy()
        del state['_metrics']
        return state

    def __setstate__(self, state: dict):
        self._metrics = {}  # populated lazily by explain()
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"Results({self.doc1_title!r} vs {self.doc2_title!r}, "
            f"{len(self.passages1)}×{len(self.passages2)} passages, "
            f"metrics={self.metrics})"
        )

    @property
    def metrics(self) -> list[str]:
        return list(self.matrices.keys())

    def top_k(
        self,
        passage: int | str,
        metric: str,
        k: int = 10,
    ) -> list[Match]:
        """Return the top-k most similar passages from the right side.

        Args:
            passage: Index (int) or label (str) of the query passage on the left side.
            metric: Name of the metric to rank by.
            k: Number of results to return.

        Returns:
            List of Match(score, index, label, text) sorted by score descending.
            Use match.index to call explain() on a specific result.
        """
        idx = self._resolve_passage(passage, self.passages1)
        scores = self.matrices[metric][idx]
        n = len(scores)
        k = min(k, n)
        if k == 0:
            return []
        # argpartition is O(n); only sort the k candidates O(k log k)
        top_indices = np.argpartition(scores, n - k)[n - k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            Match(
                score=float(scores[j]),
                index=int(j),
                label=self.passages2[j].label,
                text=self.passages2[j].text,
            )
            for j in top_indices
        ]

    def filter(
        self,
        metric: str,
        threshold: float,
    ) -> list[FilterMatch]:
        """Return all passage pairs with similarity at or above a threshold.

        Args:
            metric: Metric to filter by.
            threshold: Minimum similarity score.

        Returns:
            List of FilterMatch(i, j, score, label_i, label_j) sorted by score descending.
            label_i and label_j are None when passages have no structural labels.
        """
        matrix = self.matrices[metric]
        rows, cols = np.where(matrix >= threshold)
        results = [
            FilterMatch(
                i=int(i),
                j=int(j),
                score=float(matrix[i, j]),
                label_i=self.passages1[i].label,
                label_j=self.passages2[j].label,
            )
            for i, j in zip(rows.tolist(), cols.tolist())
        ]
        return sorted(results, key=lambda m: -m.score)

    def heatmap(
        self,
        metric: str,
        output: str | None = None,
        xlabels: bool = False,
        ylabels: bool = False,
    ):
        """Display or save a heatmap of the similarity matrix.

        Args:
            metric: Metric to visualize.
            output: File path to save the image. If None, displays interactively.
            xlabels: If True, annotate x-axis with structural labels.
            ylabels: If True, annotate y-axis with structural labels.
        """
        import matplotlib.pyplot as plt

        matrix = self.matrices[metric]
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{metric}: {self.doc1_title} vs {self.doc2_title}")
        ax.set_xlabel(self.doc2_title)
        ax.set_ylabel(self.doc1_title)

        if xlabels:
            ax.set_xticks(range(len(self.passages2)))
            ax.set_xticklabels(
                [p.label or str(p.index) for p in self.passages2],
                rotation=90, fontsize=4,
            )
        if ylabels:
            ax.set_yticks(range(len(self.passages1)))
            ax.set_yticklabels(
                [p.label or str(p.index) for p in self.passages1],
                fontsize=4,
            )

        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def diff(self, other: 'Results') -> 'Results':
        """Compute element-wise difference (self - other) for all shared metrics.

        Useful for comparing two cross-referencing runs, e.g. BoM-KJV minus BoM-JST
        to see where BoM aligns specifically with JST changes.
        """
        shared = set(self.metrics) & set(other.metrics)
        diff_matrices = {}
        for m in shared:
            if self.matrices[m].shape != other.matrices[m].shape:
                raise ValueError(
                    f"Cannot diff metric {m!r}: matrix shapes differ "
                    f"({self.matrices[m].shape} vs {other.matrices[m].shape}). "
                    "Both Results must compare the same passage sets."
                )
            diff_matrices[m] = self.matrices[m] - other.matrices[m]
        return Results(
            passages1=self.passages1,
            passages2=self.passages2,
            matrices=diff_matrices,
            doc1_title=self.doc1_title,
            doc2_title=self.doc2_title,
        )

    def save(self, path: str, backend: str = "pickle"):
        """Save results to disk.

        Args:
            path: Output file path.
            backend: Storage backend. Currently supports 'pickle'.
        """
        if backend == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(
                f"Backend {backend!r} not yet implemented. Available: 'pickle'"
            )

    @classmethod
    def load(cls, path: str) -> 'Results':
        """Load results from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def explain(
        self,
        passage1: int | str,
        passage2: int | str,
        metric: str,
    ) -> dict:
        """Return metric-specific evidence for why two passages are similar.

        Args:
            passage1: Index (int) or label (str) of the passage on the left side.
            passage2: Index (int) or label (str) of the passage on the right side.
                      Use Match.index from top_k() results directly.
            metric: Name of the metric to explain.

        Returns:
            A dict of evidence. For NGramMetric: {"score": float, "common_ngrams": {...}}.
            For other metrics: {"score": float}.

        Raises:
            KeyError: If the metric was not attached (results loaded without metric objects).
        """
        if metric not in self._metrics:
            if metric in self._metric_configs:
                from .metrics import metric_from_config
                self._metrics[metric] = metric_from_config(self._metric_configs[metric])
            else:
                raise KeyError(
                    f"Metric {metric!r} not found in this Results object. "
                    f"Available metrics: {self.metrics}"
                )
        i = self._resolve_passage(passage1, self.passages1)
        j = self._resolve_passage(passage2, self.passages2)
        return self._metrics[metric].explain(self.passages1[i].text, self.passages2[j].text)

    def _resolve_passage(self, passage: int | str, passages: list['Passage']) -> int:
        if isinstance(passage, int):
            return passage
        for i, p in enumerate(passages):
            if p.label == passage:
                return i
        raise ValueError(f"No passage with label {passage!r}")


class ResultsCollection:
    """Results from comparing multiple document pairs.

    Indexed by (i, j) integer pairs corresponding to positions in the
    input document lists passed to compare().
    """

    def __init__(
        self,
        pair_results: dict[tuple[int, int], Results],
        docs1: list,
        docs2: list,
    ):
        self._results = pair_results
        self.docs1 = docs1
        self.docs2 = docs2

    def __repr__(self) -> str:
        return f"ResultsCollection({len(self._results)} pair(s))"

    def __getitem__(self, key: tuple[int, int]) -> Results:
        return self._results[key]

    def __iter__(self):
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def items(self):
        return self._results.items()

    def pairs(self) -> Generator[tuple[str, str, Results], None, None]:
        """Iterate over all results, yielding (doc1_title, doc2_title, Results) tuples."""
        for (i, j), results in self._results.items():
            yield self.docs1[i].title, self.docs2[j].title, results

    def save(self, path: str, backend: str = "pickle"):
        """Save all results to disk."""
        if backend == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(
                f"Backend {backend!r} not yet implemented. Available: 'pickle'"
            )

    @classmethod
    def load(cls, path: str) -> 'ResultsCollection':
        """Load results from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
