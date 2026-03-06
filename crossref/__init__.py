import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .documents import Document, Passage
from .compare import compare
from .results import Results, ResultsCollection, Match, FilterMatch, CollectionMatch
from .preprocessing import generate_preprocessing_pipeline
from .metrics import (
    SimilarityMetric,
    SyntacticSimilarityMetric, NGramMetric, JaccardMetric, CharNGramMetric,
    LCSMetric, LevenshteinMetric, FuzzyNGramMetric,
    LCSubstringMetric, DiceMetric, ROUGELMetric, ROUGENMetric, PhoneticMetric,
    SemanticSimilarityMetric, SentenceTransformerMetric, CrossEncoderMetric,
    ColBERTMetric, LLMJudgeMetric, APIEmbeddingMetric,
    Doc2VecMetric, AveragedWordVecMetric, WMDMetric, SoftCosineMetric,
    StatisticalSimilarityMetric, NCDMetric, TFIDFMetric, BM25Metric,
    LSAMetric, JSMetric, LZJDMetric,
)


def load(path: str) -> Results | ResultsCollection:
    """Load saved results from disk. Backend is inferred from the file extension:
    .pkl/.pickle → pickle, .h5/.hdf5 → hdf5, .db/.sqlite → sqlite.
    """
    from .results import _infer_backend
    backend = _infer_backend(path)
    if backend == 'hdf5':
        return Results.load(path)
    if backend == 'sqlite':
        # Try Results first; fall back to ResultsCollection.
        import sqlite3
        con = sqlite3.connect(path)
        is_collection = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='doc_pairs'"
        ).fetchone() is not None
        con.close()
        return (ResultsCollection if is_collection else Results).load(path)
    with open(path, 'rb') as f:
        import pickle
        return pickle.load(f)


def main():
    from .main import main as _main
    _main()
