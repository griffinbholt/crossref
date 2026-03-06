import pickle

from .documents import Document, Passage
from .compare import compare
from .results import Results, ResultsCollection, Match, FilterMatch
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
    """Load saved results from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    from .main import main as _main
    _main()
