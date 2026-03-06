import csv
import json
import os
import pickle

import numpy as np

from typing import Generator, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .documents import Passage
    from .metrics.base import SimilarityMetric


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

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
        text_i:  Text of the left passage.
        text_j:  Text of the right passage.
    """
    i: int
    j: int
    score: float
    label_i: str | None
    label_j: str | None
    text_i: str
    text_j: str


class CollectionMatch(NamedTuple):
    """A single result from ResultsCollection.filter() — a passage pair from any document pair.

    Attributes:
        doc1_title: Title of the left document.
        doc2_title: Title of the right document.
        i:          Index of the passage in passages1 (left side).
        j:          Index of the passage in passages2 (right side).
        score:      Similarity score.
        label_i:    Structural label of the left passage, or None if unset.
        label_j:    Structural label of the right passage, or None if unset.
        text_i:     Text of the left passage.
        text_j:     Text of the right passage.
    """
    doc1_title: str
    doc2_title: str
    i: int
    j: int
    score: float
    label_i: str | None
    label_j: str | None
    text_i: str
    text_j: str


# ---------------------------------------------------------------------------
# Lazy matrix proxy for HDF5 backend
# ---------------------------------------------------------------------------

class _HDF5MatrixProxy:
    """Wraps an open h5py Dataset for lazy, row-wise access.

    Behaves like a 2D array: proxy[i] reads only row i from disk without
    loading the full matrix into RAM. Full-matrix loads (np.asarray(proxy))
    still work but pull everything into memory.
    """

    def __init__(self, dataset):
        self._ds = dataset

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._ds.shape)

    def __getitem__(self, key) -> np.ndarray:
        return np.asarray(self._ds[key])

    def __array__(self, dtype=None) -> np.ndarray:
        arr = np.asarray(self._ds[:])
        return arr if dtype is None else arr.astype(dtype)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_EXT_TO_BACKEND = {
    'pkl': 'pickle', 'pickle': 'pickle',
    'h5': 'hdf5', 'hdf5': 'hdf5',
    'db': 'sqlite', 'sqlite': 'sqlite',
}


def _infer_backend(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return _EXT_TO_BACKEND.get(ext, 'pickle')


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Results:
    """Similarity comparison results between two sets of passages.

    Stores one similarity matrix per metric, plus the passage metadata
    for both sides of the comparison.

    When loaded from an HDF5 file, matrices are backed by on-disk datasets
    and only the requested rows are read into memory (lazy access). When
    loaded from SQLite, only scores above the storage threshold are present
    (below-threshold entries are 0).
    """

    def __init__(
        self,
        passages1: list['Passage'],
        passages2: list['Passage'],
        matrices: dict[str, np.ndarray],
        doc1_title: str = "",
        doc2_title: str = "",
        metrics: dict[str, 'SimilarityMetric'] | None = None,
        preprocessing_config: list | None = None,
    ):
        self.passages1 = passages1
        self.passages2 = passages2
        self.matrices = matrices
        self.doc1_title = doc1_title
        self.doc2_title = doc2_title
        self.preprocessing_config: list | None = preprocessing_config
        self._metrics: dict[str, 'SimilarityMetric'] = metrics or {}
        self._metric_configs: dict[str, dict] = {
            name: m.config() for name, m in self._metrics.items()
        }
        self._hdf5_file = None  # set by _load_hdf5; closed on __del__ / pickle

    def __del__(self):
        if getattr(self, '_hdf5_file', None) is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state['_metrics']
        # Convert lazy proxies to numpy arrays so the pickle is self-contained.
        state['matrices'] = {
            k: np.asarray(v) if not isinstance(v, np.ndarray) else v
            for k, v in state['matrices'].items()
        }
        hdf5_file = state.pop('_hdf5_file', None)
        if hdf5_file is not None:
            try:
                hdf5_file.close()
            except Exception:
                pass
        return state

    def __setstate__(self, state: dict):
        self._metrics = {}
        self._hdf5_file = None
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

    def _matrix_as_array(self, metric: str) -> np.ndarray:
        """Return the full matrix as a numpy array, loading from disk if needed."""
        m = self.matrices[metric]
        return m if isinstance(m, np.ndarray) else np.asarray(m)

    def top_k(
        self,
        passage: int | str,
        metric: str,
        k: int | None = 10,
    ) -> list[Match]:
        """Return the top-k most similar passages from the right side.

        For HDF5-backed results, only the single requested row is read from disk.

        Args:
            passage: Index (int) or label (str) of the query passage on the left side.
            metric:  Name of the metric to rank by.
            k:       Number of results to return. Pass None to return all passages ranked.

        Returns:
            List of Match(score, index, label, text) sorted by score descending.
        """
        idx = self._resolve_passage(passage, self.passages1)
        scores = self.matrices[metric][idx]  # lazy row read for HDF5
        n = len(scores)
        if k is None or k >= n:
            top_indices = np.argsort(scores)[::-1]
        else:
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

        For HDF5-backed results, rows are read one at a time to avoid loading
        the full matrix into RAM.

        Args:
            metric:    Metric to filter by.
            threshold: Minimum similarity score.

        Returns:
            List of FilterMatch sorted by score descending.
        """
        matrix = self.matrices[metric]
        results = []
        if isinstance(matrix, np.ndarray):
            # Fast vectorized path for in-memory matrices.
            rows, cols = np.where(matrix >= threshold)
            results = [
                FilterMatch(
                    i=int(i), j=int(j),
                    score=float(matrix[i, j]),
                    label_i=self.passages1[i].label,
                    label_j=self.passages2[j].label,
                    text_i=self.passages1[i].text,
                    text_j=self.passages2[j].text,
                )
                for i, j in zip(rows.tolist(), cols.tolist())
            ]
        else:
            # Row-by-row for lazy HDF5 — reads one row at a time from disk.
            for i in range(len(self.passages1)):
                row = np.asarray(matrix[i])
                for j in np.where(row >= threshold)[0]:
                    results.append(FilterMatch(
                        i=i, j=int(j),
                        score=float(row[j]),
                        label_i=self.passages1[i].label,
                        label_j=self.passages2[j].label,
                        text_i=self.passages1[i].text,
                        text_j=self.passages2[j].text,
                    ))
        return sorted(results, key=lambda m: -m.score)

    def summary(self, metric: str) -> dict:
        """Return aggregate statistics for the similarity matrix of a given metric.

        Note: for SQLite-backed results, scores below the storage threshold
        are recorded as 0, which skews these statistics.

        Returns:
            Dict with keys: mean, median, std, min, max, shape, and
            above_0.5 / above_0.8 / above_0.9 counts.
        """
        flat = self._matrix_as_array(metric).flatten()
        return {
            'shape': self.matrices[metric].shape,
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'std': float(np.std(flat)),
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'above_0.5': int(np.sum(flat >= 0.5)),
            'above_0.8': int(np.sum(flat >= 0.8)),
            'above_0.9': int(np.sum(flat >= 0.9)),
        }

    def to_csv(self, path: str, metric: str, threshold: float | None = None):
        """Export passage pairs to a CSV file.

        Each row contains: score, label_i, label_j, text_i, text_j.
        Rows are sorted by score descending.

        Args:
            path:      Output file path.
            metric:    Metric to export scores for.
            threshold: If provided, only export pairs at or above this score.
                       If None, all pairs are exported.
        """
        matches = self.filter(metric, threshold if threshold is not None else float('-inf'))
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['score', 'label_i', 'label_j', 'text_i', 'text_j'])
            for m in matches:
                writer.writerow([m.score, m.label_i, m.label_j, m.text_i, m.text_j])

    def heatmap(
        self,
        metric: str,
        output: str | None = None,
        xlabels: bool = False,
        ylabels: bool = False,
    ):
        """Display or save a heatmap of the similarity matrix.

        For HDF5-backed results, loads the full matrix into RAM for rendering.

        Args:
            metric:  Metric to visualize.
            output:  File path to save the image. If None, displays interactively.
            xlabels: If True, annotate x-axis with structural labels.
            ylabels: If True, annotate y-axis with structural labels.
        """
        import matplotlib.pyplot as plt

        matrix = self._matrix_as_array(metric)
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
            sm = self._matrix_as_array(m)
            om = other._matrix_as_array(m)
            if sm.shape != om.shape:
                raise ValueError(
                    f"Cannot diff metric {m!r}: matrix shapes differ "
                    f"({sm.shape} vs {om.shape}). "
                    "Both Results must compare the same passage sets."
                )
            diff_matrices[m] = sm - om
        return Results(
            passages1=self.passages1,
            passages2=self.passages2,
            matrices=diff_matrices,
            doc1_title=self.doc1_title,
            doc2_title=self.doc2_title,
        )

    def save(self, path: str, backend: str | None = None, threshold: float | None = None):
        """Save results to disk.

        Args:
            path:      Output file path. Extension determines backend if backend is None:
                       .pkl/.pickle → pickle  (lossless, Python-only)
                       .h5/.hdf5    → hdf5    (lossless, gzip-compressed, lazy-loadable)
                       .db/.sqlite  → sqlite  (sparse above threshold, SQL-queryable)
            backend:   Explicit backend override: 'pickle', 'hdf5', or 'sqlite'.
            threshold: Required for SQLite — only scores at or above this value are stored.
        """
        backend = backend or _infer_backend(path)
        if backend == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif backend == 'hdf5':
            self._save_hdf5(path)
        elif backend == 'sqlite':
            if threshold is None:
                raise ValueError(
                    "SQLite backend requires a threshold (e.g. threshold=0.5). "
                    "Only scores at or above this value will be stored."
                )
            self._save_sqlite(path, threshold)
        else:
            raise NotImplementedError(
                f"Backend {backend!r} not supported. Use 'pickle', 'hdf5', or 'sqlite'."
            )

    @classmethod
    def load(cls, path: str) -> 'Results':
        """Load results from disk. Backend is inferred from the file extension."""
        backend = _infer_backend(path)
        if backend == 'hdf5':
            return cls._load_hdf5(path)
        if backend == 'sqlite':
            return cls._load_sqlite(path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_hdf5(self, path: str):
        import h5py
        with h5py.File(path, 'w') as f:
            f.attrs['doc1_title'] = self.doc1_title or ''
            f.attrs['doc2_title'] = self.doc2_title or ''
            f.attrs['preprocessing_config'] = json.dumps(self.preprocessing_config)
            f.attrs['metric_configs'] = json.dumps(self._metric_configs)
            _hdf5_write_passages(f, 'passages1', self.passages1)
            _hdf5_write_passages(f, 'passages2', self.passages2)
            mgrp = f.create_group('matrices')
            for metric, matrix in self.matrices.items():
                arr = np.asarray(matrix).astype(np.float32)
                n1, n2 = arr.shape
                # Chunk by rows for efficient row-wise reads (top_k, filter).
                chunks = (min(256, n1), n2)
                mgrp.create_dataset(
                    metric, data=arr,
                    compression='gzip', compression_opts=4,
                    chunks=chunks,
                )

    @classmethod
    def _load_hdf5(cls, path: str) -> 'Results':
        import h5py
        f = h5py.File(path, 'r')  # kept open for lazy matrix reads
        passages1 = _hdf5_read_passages(f['passages1'])
        passages2 = _hdf5_read_passages(f['passages2'])
        matrices = {
            name: _HDF5MatrixProxy(f['matrices'][name])
            for name in f['matrices']
        }
        result = cls.__new__(cls)
        result.passages1 = passages1
        result.passages2 = passages2
        result.matrices = matrices
        result.doc1_title = f.attrs.get('doc1_title', '')
        result.doc2_title = f.attrs.get('doc2_title', '')
        result.preprocessing_config = json.loads(f.attrs.get('preprocessing_config', 'null'))
        result._metrics = {}
        result._metric_configs = json.loads(f.attrs.get('metric_configs', '{}'))
        result._hdf5_file = f
        return result

    def _save_sqlite(self, path: str, threshold: float):
        import sqlite3
        con = sqlite3.connect(path)
        _sqlite_init_schema(con, collection=False)
        cur = con.cursor()
        _sqlite_write_metadata(cur, {
            'doc1_title': self.doc1_title,
            'doc2_title': self.doc2_title,
            'threshold': str(threshold),
            'preprocessing_config': json.dumps(self.preprocessing_config),
            'metric_configs': json.dumps(self._metric_configs),
        })
        _sqlite_write_passages(cur, 'passages1', self.passages1)
        _sqlite_write_passages(cur, 'passages2', self.passages2)
        for metric, matrix in self.matrices.items():
            _sqlite_write_scores(cur, metric, matrix, threshold, len(self.passages1))
        con.commit()
        con.close()

    @classmethod
    def _load_sqlite(cls, path: str) -> 'Results':
        import sqlite3
        con = sqlite3.connect(path)
        cur = con.cursor()
        meta = dict(cur.execute("SELECT key, value FROM metadata").fetchall())
        passages1 = _sqlite_read_passages(cur, 'passages1')
        passages2 = _sqlite_read_passages(cur, 'passages2')
        matrices = _sqlite_read_matrices(cur, len(passages1), len(passages2))
        con.close()
        result = cls.__new__(cls)
        result.passages1 = passages1
        result.passages2 = passages2
        result.matrices = matrices
        result.doc1_title = meta.get('doc1_title', '')
        result.doc2_title = meta.get('doc2_title', '')
        result.preprocessing_config = json.loads(meta.get('preprocessing_config', 'null'))
        result._metrics = {}
        result._metric_configs = json.loads(meta.get('metric_configs', '{}'))
        result._hdf5_file = None
        return result

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
        # Exact match first
        for i, p in enumerate(passages):
            if p.label == passage:
                return i
        # Substring match fallback
        matches = [(i, p) for i, p in enumerate(passages) if p.label and passage in p.label]
        if len(matches) == 1:
            return matches[0][0]
        if len(matches) > 1:
            labels = [p.label for _, p in matches]
            raise ValueError(
                f"Label {passage!r} is ambiguous — matches {len(matches)} passages: {labels}"
            )
        raise ValueError(f"No passage with label {passage!r}")


# ---------------------------------------------------------------------------
# ResultsCollection
# ---------------------------------------------------------------------------

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

    def filter(self, metric: str, threshold: float) -> list[CollectionMatch]:
        """Return all passage pairs across all document pairs above a similarity threshold.

        Args:
            metric:    Metric to filter by.
            threshold: Minimum similarity score.

        Returns:
            List of CollectionMatch sorted by score descending.
        """
        all_matches = []
        for results in self._results.values():
            for m in results.filter(metric, threshold):
                all_matches.append(CollectionMatch(
                    doc1_title=results.doc1_title,
                    doc2_title=results.doc2_title,
                    i=m.i,
                    j=m.j,
                    score=m.score,
                    label_i=m.label_i,
                    label_j=m.label_j,
                    text_i=m.text_i,
                    text_j=m.text_j,
                ))
        return sorted(all_matches, key=lambda m: -m.score)

    def save(self, path: str, backend: str | None = None, threshold: float | None = None):
        """Save all results to disk.

        Args:
            path:      Output file path. Extension determines backend if backend is None:
                       .pkl/.pickle → pickle  (lossless, Python-only)
                       .h5/.hdf5    → hdf5    (lossless, gzip-compressed, lazy-loadable)
                       .db/.sqlite  → sqlite  (sparse above threshold, SQL-queryable)
            backend:   Explicit backend override: 'pickle', 'hdf5', or 'sqlite'.
            threshold: Required for SQLite — only scores at or above this value are stored.
        """
        backend = backend or _infer_backend(path)
        if backend == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif backend == 'hdf5':
            self._save_hdf5(path)
        elif backend == 'sqlite':
            if threshold is None:
                raise ValueError(
                    "SQLite backend requires a threshold (e.g. threshold=0.5). "
                    "Only scores at or above this value will be stored."
                )
            self._save_sqlite(path, threshold)
        else:
            raise NotImplementedError(
                f"Backend {backend!r} not supported. Use 'pickle', 'hdf5', or 'sqlite'."
            )

    @classmethod
    def load(cls, path: str) -> 'ResultsCollection':
        """Load results from disk. Backend is inferred from the file extension."""
        backend = _infer_backend(path)
        if backend == 'hdf5':
            return cls._load_hdf5(path)
        if backend == 'sqlite':
            return cls._load_sqlite(path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_hdf5(self, path: str):
        import h5py
        with h5py.File(path, 'w') as f:
            f.attrs['type'] = 'ResultsCollection'
            for (pi, pj), results in self._results.items():
                grp = f.create_group(f'pair_{pi}_{pj}')
                grp.attrs['pair_i'] = pi
                grp.attrs['pair_j'] = pj
                grp.attrs['doc1_title'] = results.doc1_title or ''
                grp.attrs['doc2_title'] = results.doc2_title or ''
                grp.attrs['preprocessing_config'] = json.dumps(results.preprocessing_config)
                grp.attrs['metric_configs'] = json.dumps(results._metric_configs)
                _hdf5_write_passages(grp, 'passages1', results.passages1)
                _hdf5_write_passages(grp, 'passages2', results.passages2)
                mgrp = grp.create_group('matrices')
                for metric, matrix in results.matrices.items():
                    arr = np.asarray(matrix).astype(np.float32)
                    n1, n2 = arr.shape
                    chunks = (min(256, n1), n2)
                    mgrp.create_dataset(
                        metric, data=arr,
                        compression='gzip', compression_opts=4,
                        chunks=chunks,
                    )

    @classmethod
    def _load_hdf5(cls, path: str) -> 'ResultsCollection':
        import h5py
        f = h5py.File(path, 'r')
        pair_results = {}
        for key in f:
            grp = f[key]
            pi = int(grp.attrs['pair_i'])
            pj = int(grp.attrs['pair_j'])
            passages1 = _hdf5_read_passages(grp['passages1'])
            passages2 = _hdf5_read_passages(grp['passages2'])
            matrices = {
                name: _HDF5MatrixProxy(grp['matrices'][name])
                for name in grp['matrices']
            }
            result = Results.__new__(Results)
            result.passages1 = passages1
            result.passages2 = passages2
            result.matrices = matrices
            result.doc1_title = grp.attrs.get('doc1_title', '')
            result.doc2_title = grp.attrs.get('doc2_title', '')
            result.preprocessing_config = json.loads(
                grp.attrs.get('preprocessing_config', 'null')
            )
            result._metrics = {}
            result._metric_configs = json.loads(grp.attrs.get('metric_configs', '{}'))
            result._hdf5_file = f  # shared file handle; closed when collection is GC'd
            pair_results[(pi, pj)] = result

        collection = cls.__new__(cls)
        collection._results = pair_results
        collection.docs1 = []
        collection.docs2 = []
        collection._hdf5_file = f
        return collection

    def __del__(self):
        if getattr(self, '_hdf5_file', None) is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass

    def _save_sqlite(self, path: str, threshold: float):
        import sqlite3
        con = sqlite3.connect(path)
        _sqlite_init_schema(con, collection=True)
        cur = con.cursor()
        _sqlite_write_metadata(cur, {'threshold': str(threshold)})
        for (pi, pj), results in self._results.items():
            cur.execute(
                "INSERT OR REPLACE INTO doc_pairs VALUES (?, ?, ?, ?, ?, ?)",
                (pi, pj, results.doc1_title, results.doc2_title,
                 json.dumps(results.preprocessing_config),
                 json.dumps(results._metric_configs)),
            )
            _sqlite_write_passages_collection(cur, 'passages1', pi, pj, results.passages1)
            _sqlite_write_passages_collection(cur, 'passages2', pi, pj, results.passages2)
            for metric, matrix in results.matrices.items():
                _sqlite_write_scores_collection(
                    cur, pi, pj, metric, matrix, threshold, len(results.passages1)
                )
        con.commit()
        con.close()

    @classmethod
    def _load_sqlite(cls, path: str) -> 'ResultsCollection':
        import sqlite3
        con = sqlite3.connect(path)
        cur = con.cursor()
        pair_rows = cur.execute(
            "SELECT pair_i, pair_j, doc1_title, doc2_title, preprocessing_config, metric_configs "
            "FROM doc_pairs"
        ).fetchall()
        pair_results = {}
        for pi, pj, doc1_title, doc2_title, prep_cfg, metric_cfgs in pair_rows:
            passages1 = _sqlite_read_passages_collection(cur, 'passages1', pi, pj)
            passages2 = _sqlite_read_passages_collection(cur, 'passages2', pi, pj)
            matrices = _sqlite_read_matrices_collection(cur, pi, pj, len(passages1), len(passages2))
            result = Results.__new__(Results)
            result.passages1 = passages1
            result.passages2 = passages2
            result.matrices = matrices
            result.doc1_title = doc1_title or ''
            result.doc2_title = doc2_title or ''
            result.preprocessing_config = json.loads(prep_cfg or 'null')
            result._metrics = {}
            result._metric_configs = json.loads(metric_cfgs or '{}')
            result._hdf5_file = None
            pair_results[(pi, pj)] = result
        con.close()
        collection = cls.__new__(cls)
        collection._results = pair_results
        collection.docs1 = []
        collection.docs2 = []
        collection._hdf5_file = None
        return collection


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _hdf5_write_passages(parent, name: str, passages: list):
    import h5py
    grp = parent.create_group(name)
    dt = h5py.string_dtype()
    grp.create_dataset('text', data=[p.text for p in passages], dtype=dt)
    grp.create_dataset('label', data=[p.label or '' for p in passages], dtype=dt)
    grp.create_dataset('index', data=[p.index for p in passages], dtype=np.int32)


def _hdf5_read_passages(grp) -> list:
    from .documents import Passage
    texts = grp['text'].asstr()[:]
    labels = grp['label'].asstr()[:]
    indices = grp['index'][:]
    return [
        Passage(text=str(t), index=int(idx), label=str(l) if l else None)
        for t, l, idx in zip(texts, labels, indices)
    ]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _sqlite_init_schema(con, collection: bool = False):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
    """)
    if collection:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS doc_pairs (
                pair_i INTEGER, pair_j INTEGER,
                doc1_title TEXT, doc2_title TEXT,
                preprocessing_config TEXT, metric_configs TEXT,
                PRIMARY KEY (pair_i, pair_j)
            );
            CREATE TABLE IF NOT EXISTS passages1 (
                pair_i INTEGER, pair_j INTEGER, i INTEGER, label TEXT, text TEXT,
                PRIMARY KEY (pair_i, pair_j, i)
            );
            CREATE TABLE IF NOT EXISTS passages2 (
                pair_i INTEGER, pair_j INTEGER, i INTEGER, label TEXT, text TEXT,
                PRIMARY KEY (pair_i, pair_j, i)
            );
            CREATE TABLE IF NOT EXISTS scores (
                pair_i INTEGER, pair_j INTEGER, metric TEXT,
                i INTEGER, j INTEGER, score REAL,
                PRIMARY KEY (pair_i, pair_j, metric, i, j)
            );
            CREATE INDEX IF NOT EXISTS idx_scores_metric_score
                ON scores(pair_i, pair_j, metric, score DESC);
            CREATE INDEX IF NOT EXISTS idx_scores_i
                ON scores(pair_i, pair_j, metric, i, score DESC);
        """)
    else:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS passages1 (
                i INTEGER PRIMARY KEY, label TEXT, text TEXT
            );
            CREATE TABLE IF NOT EXISTS passages2 (
                i INTEGER PRIMARY KEY, label TEXT, text TEXT
            );
            CREATE TABLE IF NOT EXISTS scores (
                metric TEXT, i INTEGER, j INTEGER, score REAL,
                PRIMARY KEY (metric, i, j)
            );
            CREATE INDEX IF NOT EXISTS idx_scores_metric_score
                ON scores(metric, score DESC);
            CREATE INDEX IF NOT EXISTS idx_scores_i
                ON scores(metric, i, score DESC);
        """)


def _sqlite_write_metadata(cur, meta: dict):
    cur.executemany(
        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
        list(meta.items()),
    )


def _sqlite_write_passages(cur, table: str, passages: list):
    cur.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)",
        [(p.index, p.label, p.text) for p in passages],
    )


def _sqlite_write_scores(cur, metric: str, matrix, threshold: float, n1: int):
    rows = []
    if isinstance(matrix, np.ndarray):
        is_, js_ = np.where(matrix >= threshold)
        for i, j in zip(is_.tolist(), js_.tolist()):
            rows.append((metric, int(i), int(j), float(matrix[i, j])))
    else:
        for i in range(n1):
            row = np.asarray(matrix[i])
            for j in np.where(row >= threshold)[0]:
                rows.append((metric, int(i), int(j), float(row[j])))
    cur.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)", rows)


def _sqlite_read_passages(cur, table: str) -> list:
    from .documents import Passage
    return [
        Passage(text=text, index=i, label=label)
        for i, label, text in cur.execute(f"SELECT i, label, text FROM {table} ORDER BY i")
    ]


def _sqlite_read_matrices(cur, n1: int, n2: int) -> dict[str, np.ndarray]:
    metric_names = [r[0] for r in cur.execute("SELECT DISTINCT metric FROM scores")]
    matrices = {}
    for metric in metric_names:
        mat = np.zeros((n1, n2), dtype=np.float32)
        for i, j, score in cur.execute(
            "SELECT i, j, score FROM scores WHERE metric=?", (metric,)
        ):
            mat[i, j] = score
        matrices[metric] = mat
    return matrices


def _sqlite_write_passages_collection(cur, table: str, pi: int, pj: int, passages: list):
    cur.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?)",
        [(pi, pj, p.index, p.label, p.text) for p in passages],
    )


def _sqlite_write_scores_collection(
    cur, pi: int, pj: int, metric: str, matrix, threshold: float, n1: int
):
    rows = []
    if isinstance(matrix, np.ndarray):
        is_, js_ = np.where(matrix >= threshold)
        for i, j in zip(is_.tolist(), js_.tolist()):
            rows.append((pi, pj, metric, int(i), int(j), float(matrix[i, j])))
    else:
        for i in range(n1):
            row = np.asarray(matrix[i])
            for j in np.where(row >= threshold)[0]:
                rows.append((pi, pj, metric, int(i), int(j), float(row[j])))
    cur.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?)", rows)


def _sqlite_read_passages_collection(cur, table: str, pi: int, pj: int) -> list:
    from .documents import Passage
    return [
        Passage(text=text, index=i, label=label)
        for i, label, text in cur.execute(
            f"SELECT i, label, text FROM {table} WHERE pair_i=? AND pair_j=? ORDER BY i",
            (pi, pj),
        )
    ]


def _sqlite_read_matrices_collection(
    cur, pi: int, pj: int, n1: int, n2: int
) -> dict[str, np.ndarray]:
    metric_names = [
        r[0] for r in cur.execute(
            "SELECT DISTINCT metric FROM scores WHERE pair_i=? AND pair_j=?", (pi, pj)
        )
    ]
    matrices = {}
    for metric in metric_names:
        mat = np.zeros((n1, n2), dtype=np.float32)
        for i, j, score in cur.execute(
            "SELECT i, j, score FROM scores WHERE pair_i=? AND pair_j=? AND metric=?",
            (pi, pj, metric),
        ):
            mat[i, j] = score
        matrices[metric] = mat
    return matrices
