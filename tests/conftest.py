"""Shared fixtures for all tests."""
import nltk
import numpy as np
import pytest

from crossref.documents import Document, Passage
from crossref.metrics.syntactic import JaccardMetric
from crossref.results import Results, ResultsCollection


@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    """Download required NLTK data once per test session."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)


@pytest.fixture
def metric():
    """A dependency-free metric for use across tests."""
    return JaccardMetric()


@pytest.fixture
def doc_a():
    return Document.from_passages(
        ["the quick brown fox", "jumps over the lazy dog", "hello world"],
        labels=["a:1", "a:2", "a:3"],
        title="Doc A",
    )


@pytest.fixture
def doc_b():
    return Document.from_passages(
        ["the quick brown fox", "a completely different passage", "over the lazy dog"],
        labels=["b:1", "b:2", "b:3"],
        title="Doc B",
    )


@pytest.fixture
def simple_results(doc_a, doc_b, metric):
    matrix = metric.score_all(doc_a.texts(), doc_b.texts())
    return Results(
        passages1=doc_a.passages,
        passages2=doc_b.passages,
        matrices={"jaccard": matrix},
        doc1_title="Doc A",
        doc2_title="Doc B",
        metrics={"jaccard": metric},
    )


@pytest.fixture
def simple_collection(doc_a, doc_b, metric):
    m_ab = metric.score_all(doc_a.texts(), doc_b.texts())
    m_ba = metric.score_all(doc_b.texts(), doc_a.texts())
    r_ab = Results(
        passages1=doc_a.passages, passages2=doc_b.passages,
        matrices={"jaccard": m_ab}, doc1_title="Doc A", doc2_title="Doc B",
        metrics={"jaccard": metric},
    )
    r_ba = Results(
        passages1=doc_b.passages, passages2=doc_a.passages,
        matrices={"jaccard": m_ba}, doc1_title="Doc B", doc2_title="Doc A",
        metrics={"jaccard": metric},
    )
    return ResultsCollection({(0, 0): r_ab, (1, 0): r_ba}, [doc_a, doc_b], [doc_a])
