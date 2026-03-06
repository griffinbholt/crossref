from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .documents import Document
from .metrics.base import SimilarityMetric
from .results import Results, ResultsCollection


def compare(
    docs1: Document | list[Document],
    docs2: Document | list[Document] | None = None,
    metrics: list[SimilarityMetric] | None = None,
    preprocess_fn: Callable[[str], str] | None = None,
    preprocessing_config: list | None = None,
    workers: int = 1,
) -> Results | ResultsCollection:
    """Compare documents using the provided metrics.

    Supports all comparison topologies:
      - Single vs. single:  compare(doc_a, doc_b, metrics)
      - Self-comparison:    compare(doc_a, metrics)
      - List vs. list:      compare([doc_a, doc_b], [doc_c, doc_d], metrics)
      - All-vs-all:         compare([doc_a, doc_b, doc_c], metrics)

    Args:
        docs1: A Document or list of Documents (left side).
        docs2: A Document, list of Documents (right side), or None for
               self-comparison of docs1 against itself.
        metrics: List of SimilarityMetric instances to compute.
        preprocess_fn: Optional function applied to each passage text before
                       scoring. Raw text in Passage.text is never modified;
                       preprocessing only affects what is passed to metrics.
        preprocessing_config: Raw preprocessing config (e.g. from YAML) stored
                              in Results for reproducibility. Not used for scoring.
        workers: Number of threads for parallel pair scoring. Each pair runs all
                 its metrics sequentially in one thread, so the same metric object
                 is never called concurrently. Safe to combine with metric-level
                 process pools (e.g. NGramMetric(workers=N)), but note that total
                 process count will be compare workers × metric workers — prefer
                 keeping one of them at 1.

    Returns:
        A Results object for single-pair comparisons, or a ResultsCollection
        (indexed by (i, j) int pairs) for multi-document comparisons.
    """
    if not metrics:
        raise ValueError("At least one metric must be provided")

    single_input = isinstance(docs1, Document)
    docs1_list: list[Document] = [docs1] if single_input else list(docs1)

    self_compare = docs2 is None
    if self_compare:
        docs2_list = docs1_list
    elif isinstance(docs2, Document):
        docs2_list = [docs2]
    else:
        docs2_list = list(docs2)

    single_output = single_input and (self_compare or len(docs2_list) == 1)

    def _preprocess(texts: list[str]) -> list[str]:
        if preprocess_fn is None:
            return texts
        return [preprocess_fn(t) for t in texts]

    def _score_pair(args: tuple[int, int, Document, Document]) -> tuple[tuple[int, int], Results]:
        i, j, doc1, doc2 = args
        if doc1 is doc2:
            texts = _preprocess(doc1.texts())
            matrices = {m.name: m.score_self(texts) for m in metrics}
        else:
            texts1 = _preprocess(doc1.texts())
            texts2 = _preprocess(doc2.texts())
            matrices = {m.name: m.score_all(texts1, texts2) for m in metrics}
        return (i, j), Results(
            passages1=doc1.passages, passages2=doc2.passages,
            matrices=matrices, doc1_title=doc1.title, doc2_title=doc2.title,
            metrics={m.name: m for m in metrics},
            preprocessing_config=preprocessing_config,
        )

    pairs = [
        (i, j, doc1, doc2)
        for i, doc1 in enumerate(docs1_list)
        for j, doc2 in enumerate(docs2_list)
    ]

    pair_results: dict[tuple[int, int], Results] = {}
    if workers == 1 or len(pairs) == 1:
        for args in pairs:
            key, result = _score_pair(args)
            pair_results[key] = result
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as ex:
            for key, result in ex.map(_score_pair, pairs):
                pair_results[key] = result

    if single_output:
        return pair_results[(0, 0)]
    return ResultsCollection(pair_results, docs1_list, docs2_list)
