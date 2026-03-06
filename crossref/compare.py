from .documents import Document
from .metrics.base import SimilarityMetric
from .results import Results, ResultsCollection


def compare(
    docs1: Document | list[Document],
    docs2: Document | list[Document] | None = None,
    metrics: list[SimilarityMetric] | None = None,
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

    pair_results: dict[tuple[int, int], Results] = {}
    for i, doc1 in enumerate(docs1_list):
        for j, doc2 in enumerate(docs2_list):
            if doc1 is doc2:
                texts = doc1.texts()
                matrices = {metric.name: metric.score_self(texts) for metric in metrics}
            else:
                matrices = {metric.name: metric.score_all(doc1.texts(), doc2.texts()) for metric in metrics}
            pair_results[(i, j)] = Results(
                passages1=doc1.passages, passages2=doc2.passages,
                matrices=matrices, doc1_title=doc1.title, doc2_title=doc2.title,
                metrics={metric.name: metric for metric in metrics},
            )

    if single_output:
        return pair_results[(0, 0)]
    return ResultsCollection(pair_results, docs1_list, docs2_list)
