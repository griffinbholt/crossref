"""Tests for crossref.compare."""
import numpy as np
import pytest

from crossref.compare import compare
from crossref.documents import Document
from crossref.metrics.syntactic import JaccardMetric, LCSMetric
from crossref.results import Results, ResultsCollection


@pytest.fixture
def jaccard():
    return JaccardMetric()


@pytest.fixture
def lcs():
    return LCSMetric()


@pytest.fixture
def doc_x():
    return Document.from_passages(
        ["the quick brown fox", "jumps over the lazy dog"],
        labels=["x:1", "x:2"],
        title="Doc X",
    )


@pytest.fixture
def doc_y():
    return Document.from_passages(
        ["the quick cat", "sits on a mat"],
        labels=["y:1", "y:2"],
        title="Doc Y",
    )


@pytest.fixture
def doc_z():
    return Document.from_passages(
        ["something completely different", "and another thing"],
        title="Doc Z",
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestCompareValidation:
    def test_no_metrics_raises(self, doc_x, doc_y):
        with pytest.raises(ValueError, match="metric"):
            compare(doc_x, doc_y, metrics=[])

    def test_none_metrics_raises(self, doc_x, doc_y):
        with pytest.raises(ValueError, match="metric"):
            compare(doc_x, doc_y, metrics=None)


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------

class TestCompareTopologies:
    def test_single_vs_single_returns_results(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert isinstance(result, Results)

    def test_self_compare_returns_results(self, doc_x, jaccard):
        result = compare(doc_x, metrics=[jaccard])
        assert isinstance(result, Results)

    def test_list_vs_list_returns_collection(self, doc_x, doc_y, doc_z, jaccard):
        result = compare([doc_x, doc_y], [doc_y, doc_z], metrics=[jaccard])
        assert isinstance(result, ResultsCollection)

    def test_all_vs_all_list_returns_collection(self, doc_x, doc_y, jaccard):
        result = compare([doc_x, doc_y], metrics=[jaccard])
        assert isinstance(result, ResultsCollection)

    def test_single_doc_single_doc2_list_one_returns_results(self, doc_x, doc_y, jaccard):
        # docs1 is a single Document, docs2 is a single Document → Results
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert isinstance(result, Results)

    def test_list_of_one_vs_one_returns_collection(self, doc_x, doc_y, jaccard):
        result = compare([doc_x], [doc_y], metrics=[jaccard])
        assert isinstance(result, ResultsCollection)

    def test_all_vs_all_pair_count(self, doc_x, doc_y, doc_z, jaccard):
        result = compare([doc_x, doc_y, doc_z], metrics=[jaccard])
        # 3 docs → 3×3 = 9 pairs
        assert len(result) == 9


# ---------------------------------------------------------------------------
# Matrix shape and content
# ---------------------------------------------------------------------------

class TestCompareMatrices:
    def test_matrix_shape_pair(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        mat = result.matrices["jaccard"]
        assert mat.shape == (len(doc_x), len(doc_y))

    def test_self_compare_matrix_square(self, doc_x, jaccard):
        result = compare(doc_x, metrics=[jaccard])
        mat = result.matrices["jaccard"]
        assert mat.shape == (len(doc_x), len(doc_x))

    def test_self_compare_diagonal_high(self, doc_x, jaccard):
        result = compare(doc_x, metrics=[jaccard])
        mat = result.matrices["jaccard"]
        for i in range(len(doc_x)):
            assert mat[i, i] == pytest.approx(1.0)

    def test_multiple_metrics(self, doc_x, doc_y, jaccard, lcs):
        result = compare(doc_x, doc_y, metrics=[jaccard, lcs])
        assert "jaccard" in result.matrices
        assert "lcs" in result.matrices

    def test_scores_in_range(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        mat = result.matrices["jaccard"]
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Passages and metadata
# ---------------------------------------------------------------------------

class TestComparePassages:
    def test_passages1_preserved(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert len(result.passages1) == len(doc_x)
        assert result.passages1[0].text == doc_x[0].text

    def test_passages2_preserved(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert len(result.passages2) == len(doc_y)

    def test_doc_titles_preserved(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert result.doc1_title == "Doc X"
        assert result.doc2_title == "Doc Y"

    def test_raw_passages_not_preprocessed(self, doc_x, doc_y, jaccard):
        def fn(t): return t.upper()
        result = compare(doc_x, doc_y, metrics=[jaccard], preprocess_fn=fn)
        # Raw text in passages should be unchanged
        assert result.passages1[0].text == doc_x[0].text
        assert result.passages2[0].text == doc_y[0].text


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestComparePreprocessing:
    def test_preprocess_fn_applied_to_scoring(self):
        # NCDMetric is case-sensitive — lowercasing identical texts raises its score.
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric()
        text = "hello world " * 20
        doc1 = Document.from_passages([text.upper()])
        doc2 = Document.from_passages([text])
        result_no_pre = compare(doc1, doc2, metrics=[m])
        result_with_pre = compare(doc1, doc2, metrics=[m], preprocess_fn=str.lower)
        # After lowercasing, both texts are identical → NCD similarity is higher
        assert result_with_pre.matrices["ncd"][0, 0] >= result_no_pre.matrices["ncd"][0, 0]

    def test_preprocessing_config_stored(self, doc_x, doc_y, jaccard):
        config = [{"normalize_text": True}]
        result = compare(doc_x, doc_y, metrics=[jaccard], preprocessing_config=config)
        assert result.preprocessing_config == config

    def test_preprocessing_config_none_by_default(self, doc_x, doc_y, jaccard):
        result = compare(doc_x, doc_y, metrics=[jaccard])
        assert result.preprocessing_config is None


# ---------------------------------------------------------------------------
# ResultsCollection indexing
# ---------------------------------------------------------------------------

class TestCollectionIndexing:
    def test_getitem(self, doc_x, doc_y, doc_z, jaccard):
        result = compare([doc_x, doc_y], [doc_z], metrics=[jaccard])
        assert isinstance(result[0, 0], Results)
        assert isinstance(result[1, 0], Results)

    def test_pairs_yields_titles(self, doc_x, doc_y, jaccard):
        result = compare([doc_x], [doc_y], metrics=[jaccard])
        pairs = list(result.pairs())
        assert len(pairs) == 1
        t1, t2, r = pairs[0]
        assert t1 == "Doc X"
        assert t2 == "Doc Y"

    def test_iter_keys(self, doc_x, doc_y, doc_z, jaccard):
        result = compare([doc_x, doc_y], [doc_z], metrics=[jaccard])
        keys = list(result)
        assert (0, 0) in keys
        assert (1, 0) in keys

    def test_self_compare_uses_score_self(self, doc_x, jaccard):
        # When doc is the same object, compare() uses score_self, not score_all
        result = compare(doc_x, metrics=[jaccard])
        # Diagonal should still be 1.0 from score_self
        mat = result.matrices["jaccard"]
        for i in range(len(doc_x)):
            assert mat[i, i] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------

class TestCompareWorkers:
    def test_workers_matches_sequential(self, doc_x, doc_y, doc_z, jaccard):
        """Parallel results should be numerically identical to sequential."""
        result_seq = compare([doc_x, doc_y], [doc_z], metrics=[jaccard], workers=1)
        result_par = compare([doc_x, doc_y], [doc_z], metrics=[jaccard], workers=2)
        for key in result_seq:
            seq_mat = result_seq[key].matrices["jaccard"]
            par_mat = result_par[key].matrices["jaccard"]
            assert np.allclose(seq_mat, par_mat)

    def test_workers_greater_than_pairs_clamps(self, doc_x, doc_y, jaccard):
        """workers > num_pairs should not raise."""
        result = compare([doc_x], [doc_y], metrics=[jaccard], workers=100)
        assert isinstance(result, ResultsCollection)

    def test_workers_single_pair_returns_results(self, doc_x, doc_y, jaccard):
        """Single pair always returns Results regardless of workers."""
        result = compare(doc_x, doc_y, metrics=[jaccard], workers=4)
        assert isinstance(result, Results)
