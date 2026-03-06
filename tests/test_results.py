"""Tests for crossref.results — Results, ResultsCollection, and storage backends."""
import pickle

import numpy as np
import pytest

from crossref.documents import Document, Passage
from crossref.metrics.syntactic import JaccardMetric
from crossref.results import (
    Results, ResultsCollection,
    Match, FilterMatch, CollectionMatch,
)


# ---------------------------------------------------------------------------
# NamedTuple field checks
# ---------------------------------------------------------------------------

class TestNamedTuples:
    def test_match_fields(self):
        m = Match(score=0.9, index=2, label="L", text="hello")
        assert m.score == 0.9
        assert m.index == 2
        assert m.label == "L"
        assert m.text == "hello"

    def test_match_label_none(self):
        m = Match(score=0.5, index=0, label=None, text="x")
        assert m.label is None

    def test_filter_match_fields(self):
        m = FilterMatch(i=0, j=1, score=0.8, label_i="A", label_j="B", text_i="tx", text_j="ty")
        assert m.i == 0 and m.j == 1
        assert m.score == 0.8
        assert m.text_i == "tx" and m.text_j == "ty"

    def test_collection_match_fields(self):
        m = CollectionMatch(
            doc1_title="D1", doc2_title="D2",
            i=0, j=1, score=0.7,
            label_i="L1", label_j="L2",
            text_i="t1", text_j="t2",
        )
        assert m.doc1_title == "D1"
        assert m.doc2_title == "D2"
        assert m.score == 0.7


# ---------------------------------------------------------------------------
# Results basics
# ---------------------------------------------------------------------------

class TestResultsBasics:
    def test_metrics_property(self, simple_results):
        assert simple_results.metrics == ["jaccard"]

    def test_repr(self, simple_results):
        r = repr(simple_results)
        assert "Doc A" in r
        assert "Doc B" in r

    def test_passages_preserved(self, simple_results, doc_a, doc_b):
        assert len(simple_results.passages1) == len(doc_a)
        assert len(simple_results.passages2) == len(doc_b)

    def test_matrix_shape(self, simple_results, doc_a, doc_b):
        mat = simple_results.matrices["jaccard"]
        assert mat.shape == (len(doc_a), len(doc_b))

    def test_preprocessing_config_none(self, simple_results):
        assert simple_results.preprocessing_config is None

    def test_preprocessing_config_stored(self, doc_a, doc_b, metric):
        cfg = [{"normalize_text": True}]
        matrix = metric.score_all(doc_a.texts(), doc_b.texts())
        r = Results(
            passages1=doc_a.passages,
            passages2=doc_b.passages,
            matrices={"jaccard": matrix},
            preprocessing_config=cfg,
        )
        assert r.preprocessing_config == cfg


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

class TestTopK:
    def test_returns_k_results(self, simple_results):
        results = simple_results.top_k(0, "jaccard", k=2)
        assert len(results) == 2

    def test_sorted_descending(self, simple_results):
        results = simple_results.top_k(0, "jaccard", k=3)
        scores = [m.score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_match_has_text(self, simple_results):
        results = simple_results.top_k(0, "jaccard", k=1)
        assert isinstance(results[0].text, str)
        assert len(results[0].text) > 0

    def test_match_has_index(self, simple_results):
        results = simple_results.top_k(0, "jaccard", k=1)
        assert isinstance(results[0].index, int)

    def test_k_none_returns_all(self, simple_results, doc_b):
        results = simple_results.top_k(0, "jaccard", k=None)
        assert len(results) == len(doc_b)

    def test_k_zero_returns_empty(self, simple_results):
        results = simple_results.top_k(0, "jaccard", k=0)
        assert results == []

    def test_k_larger_than_n_returns_all(self, simple_results, doc_b):
        results = simple_results.top_k(0, "jaccard", k=1000)
        assert len(results) == len(doc_b)

    def test_label_lookup_exact(self, simple_results):
        results_by_idx = simple_results.top_k(0, "jaccard", k=3)
        results_by_label = simple_results.top_k("a:1", "jaccard", k=3)
        assert [m.score for m in results_by_idx] == [m.score for m in results_by_label]

    def test_label_lookup_substring(self, simple_results):
        # "a:" should match "a:1" (only one match)
        results = simple_results.top_k("a:1", "jaccard", k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

class TestFilter:
    def test_all_above_zero(self, simple_results):
        results = simple_results.filter("jaccard", 0.0)
        assert len(results) > 0

    def test_threshold_filters(self, simple_results):
        all_results = simple_results.filter("jaccard", 0.0)
        high_results = simple_results.filter("jaccard", 0.9)
        assert len(high_results) <= len(all_results)

    def test_sorted_descending(self, simple_results):
        results = simple_results.filter("jaccard", 0.0)
        scores = [m.score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_match_has_texts(self, simple_results):
        results = simple_results.filter("jaccard", 0.0)
        for m in results:
            assert isinstance(m.text_i, str)
            assert isinstance(m.text_j, str)

    def test_match_has_labels(self, simple_results):
        results = simple_results.filter("jaccard", 0.0)
        for m in results:
            assert m.label_i is not None  # doc_a has labels
            assert m.label_j is not None  # doc_b has labels

    def test_above_one_returns_empty(self, simple_results):
        results = simple_results.filter("jaccard", 2.0)
        assert results == []

    def test_filter_match_indices_valid(self, simple_results, doc_a, doc_b):
        results = simple_results.filter("jaccard", 0.0)
        for m in results:
            assert 0 <= m.i < len(doc_a)
            assert 0 <= m.j < len(doc_b)

    def test_scores_match_matrix(self, simple_results):
        results = simple_results.filter("jaccard", 0.0)
        mat = simple_results.matrices["jaccard"]
        for m in results:
            assert mat[m.i, m.j] == pytest.approx(m.score, abs=1e-5)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_returns_dict(self, simple_results):
        s = simple_results.summary("jaccard")
        assert isinstance(s, dict)

    def test_has_required_keys(self, simple_results):
        s = simple_results.summary("jaccard")
        for key in ("mean", "median", "std", "min", "max", "shape", "above_0.5", "above_0.8", "above_0.9"):
            assert key in s, f"Missing key: {key}"

    def test_shape_correct(self, simple_results, doc_a, doc_b):
        s = simple_results.summary("jaccard")
        assert s["shape"] == (len(doc_a), len(doc_b))

    def test_above_counts_consistent(self, simple_results):
        s = simple_results.summary("jaccard")
        assert s["above_0.9"] <= s["above_0.8"] <= s["above_0.5"]

    def test_min_max_range(self, simple_results):
        s = simple_results.summary("jaccard")
        assert s["min"] <= s["mean"] <= s["max"]


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

class TestToCSV:
    def test_creates_file(self, simple_results, tmp_path):
        path = str(tmp_path / "out.csv")
        simple_results.to_csv(path, "jaccard")
        import os
        assert os.path.exists(path)

    def test_header_row(self, simple_results, tmp_path):
        import csv
        path = str(tmp_path / "out.csv")
        simple_results.to_csv(path, "jaccard")
        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) >= {"score", "label_i", "label_j", "text_i", "text_j"}

    def test_row_count_without_threshold(self, simple_results, tmp_path, doc_a, doc_b):
        import csv
        path = str(tmp_path / "out.csv")
        simple_results.to_csv(path, "jaccard")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(doc_a) * len(doc_b)

    def test_threshold_reduces_rows(self, simple_results, tmp_path):
        import csv
        path_all = str(tmp_path / "all.csv")
        path_thr = str(tmp_path / "thr.csv")
        simple_results.to_csv(path_all, "jaccard")
        simple_results.to_csv(path_thr, "jaccard", threshold=0.5)
        with open(path_all) as f:
            all_rows = list(csv.DictReader(f))
        with open(path_thr) as f:
            thr_rows = list(csv.DictReader(f))
        assert len(thr_rows) <= len(all_rows)

    def test_sorted_descending(self, simple_results, tmp_path):
        import csv
        path = str(tmp_path / "out.csv")
        simple_results.to_csv(path, "jaccard")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        scores = [float(r["score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_same_is_zero(self, simple_results):
        diff = simple_results.diff(simple_results)
        mat = diff.matrices["jaccard"]
        assert np.allclose(mat, 0.0)

    def test_diff_shape_preserved(self, simple_results):
        diff = simple_results.diff(simple_results)
        assert diff.matrices["jaccard"].shape == simple_results.matrices["jaccard"].shape

    def test_diff_only_shared_metrics(self, doc_a, doc_b, metric):
        lcs = __import__("crossref.metrics.syntactic", fromlist=["LCSMetric"]).LCSMetric()
        mat_j = metric.score_all(doc_a.texts(), doc_b.texts())
        mat_l = lcs.score_all(doc_a.texts(), doc_b.texts())
        r1 = Results(doc_a.passages, doc_b.passages, {"jaccard": mat_j})
        r2 = Results(doc_a.passages, doc_b.passages, {"lcs": mat_l})
        diff = r1.diff(r2)
        assert diff.metrics == []  # no shared metrics

    def test_diff_shape_mismatch_raises(self, doc_a, doc_b, doc_a_doc_b=None):
        from crossref.metrics.syntactic import JaccardMetric
        m = JaccardMetric()
        r1 = Results(
            doc_a.passages, doc_b.passages,
            {"jaccard": m.score_all(doc_a.texts(), doc_b.texts())},
        )
        doc_c = Document.from_passages(["x", "y", "z", "w"])
        r2 = Results(
            doc_a.passages, doc_c.passages,
            {"jaccard": m.score_all(doc_a.texts(), doc_c.texts())},
        )
        with pytest.raises(ValueError, match="shape"):
            r1.diff(r2)


# ---------------------------------------------------------------------------
# _resolve_passage
# ---------------------------------------------------------------------------

class TestResolvePassage:
    def test_int_index(self, simple_results):
        # _resolve_passage is called internally; test via top_k
        r = simple_results.top_k(0, "jaccard", k=1)
        assert len(r) == 1

    def test_exact_label(self, simple_results):
        r = simple_results.top_k("a:1", "jaccard", k=1)
        assert len(r) == 1

    def test_substring_match_single(self, simple_results):
        # "a:2" is unique in doc_a labels
        r = simple_results.top_k("a:2", "jaccard", k=1)
        assert len(r) == 1

    def test_not_found_raises(self, simple_results):
        with pytest.raises(ValueError, match="No passage"):
            simple_results.top_k("nonexistent:label", "jaccard", k=1)

    def test_ambiguous_raises(self):
        # Create doc where "a:" matches multiple labels
        doc = Document.from_passages(["x", "y"], labels=["a:1 foo", "a:1 bar"])
        m = JaccardMetric()
        mat = m.score_all(doc.texts(), doc.texts())
        r = Results(doc.passages, doc.passages, {"jaccard": mat})
        with pytest.raises(ValueError, match="ambiguous"):
            r.top_k("a:1", "jaccard", k=1)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_returns_score(self, simple_results):
        ev = simple_results.explain(0, 0, "jaccard")
        assert "score" in ev

    def test_explain_by_label(self, simple_results):
        ev = simple_results.explain("a:1", "b:1", "jaccard")
        assert "score" in ev

    def test_explain_unknown_metric_raises(self, simple_results):
        with pytest.raises(KeyError, match="bogus"):
            simple_results.explain(0, 0, "bogus")

    def test_explain_reconstructs_metric_from_config(self, doc_a, doc_b, metric):
        """Metric should be reconstructable from stored config after pickling."""
        matrix = metric.score_all(doc_a.texts(), doc_b.texts())
        r = Results(
            passages1=doc_a.passages,
            passages2=doc_b.passages,
            matrices={"jaccard": matrix},
            doc1_title="A", doc2_title="B",
            metrics={"jaccard": metric},
        )
        # Simulate pickle roundtrip (drops live metric objects)
        import pickle
        r2 = pickle.loads(pickle.dumps(r))
        ev = r2.explain(0, 0, "jaccard")
        assert "score" in ev


# ---------------------------------------------------------------------------
# Pickle backend
# ---------------------------------------------------------------------------

class TestPickleBackend:
    def test_save_and_load(self, simple_results, tmp_path):
        path = str(tmp_path / "results.pkl")
        simple_results.save(path)
        loaded = Results.load(path)
        assert loaded.doc1_title == simple_results.doc1_title
        assert loaded.doc2_title == simple_results.doc2_title

    def test_matrices_preserved(self, simple_results, tmp_path):
        path = str(tmp_path / "results.pkl")
        simple_results.save(path)
        loaded = Results.load(path)
        assert np.allclose(loaded.matrices["jaccard"], simple_results.matrices["jaccard"])

    def test_passages_preserved(self, simple_results, tmp_path):
        path = str(tmp_path / "results.pkl")
        simple_results.save(path)
        loaded = Results.load(path)
        assert [p.text for p in loaded.passages1] == [p.text for p in simple_results.passages1]

    def test_backend_inferred_from_extension(self, simple_results, tmp_path):
        path = str(tmp_path / "results.pickle")
        simple_results.save(path)
        loaded = Results.load(path)
        assert isinstance(loaded, Results)

    def test_preprocessing_config_preserved(self, doc_a, doc_b, metric, tmp_path):
        cfg = [{"normalize_text": True}]
        matrix = metric.score_all(doc_a.texts(), doc_b.texts())
        r = Results(
            passages1=doc_a.passages, passages2=doc_b.passages,
            matrices={"jaccard": matrix}, preprocessing_config=cfg,
        )
        path = str(tmp_path / "r.pkl")
        r.save(path)
        loaded = Results.load(path)
        assert loaded.preprocessing_config == cfg


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class TestSQLiteBackend:
    def test_save_requires_threshold(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        with pytest.raises(ValueError, match="threshold"):
            simple_results.save(path, backend="sqlite")

    def test_save_and_load(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        simple_results.save(path, threshold=0.0)
        loaded = Results.load(path)
        assert loaded.doc1_title == simple_results.doc1_title

    def test_passages_preserved(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        simple_results.save(path, threshold=0.0)
        loaded = Results.load(path)
        assert [p.text for p in loaded.passages1] == [p.text for p in simple_results.passages1]
        assert [p.label for p in loaded.passages1] == [p.label for p in simple_results.passages1]

    def test_scores_above_threshold_stored(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        threshold = 0.5
        simple_results.save(path, threshold=threshold)
        loaded = Results.load(path)
        mat = loaded.matrices["jaccard"]
        orig_mat = simple_results.matrices["jaccard"]
        # Scores above threshold should match
        mask = orig_mat >= threshold
        assert np.allclose(mat[mask], orig_mat[mask], atol=1e-5)

    def test_scores_below_threshold_are_zero(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        threshold = 0.9
        simple_results.save(path, threshold=threshold)
        loaded = Results.load(path)
        mat = loaded.matrices["jaccard"]
        orig_mat = simple_results.matrices["jaccard"]
        below_mask = orig_mat < threshold
        assert np.all(mat[below_mask] == 0.0)

    def test_filter_works_after_load(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        simple_results.save(path, threshold=0.0)
        loaded = Results.load(path)
        matches = loaded.filter("jaccard", 0.0)
        assert len(matches) > 0

    def test_top_k_works_after_load(self, simple_results, tmp_path):
        path = str(tmp_path / "r.db")
        simple_results.save(path, threshold=0.0)
        loaded = Results.load(path)
        matches = loaded.top_k(0, "jaccard", k=2)
        assert len(matches) == 2

    def test_extension_inferred(self, simple_results, tmp_path):
        path = str(tmp_path / "r.sqlite")
        simple_results.save(path, threshold=0.0)
        loaded = Results.load(path)
        assert isinstance(loaded, Results)


# ---------------------------------------------------------------------------
# HDF5 backend
# ---------------------------------------------------------------------------

class TestHDF5Backend:
    @pytest.fixture(autouse=True)
    def require_h5py(self):
        pytest.importorskip("h5py")

    def test_save_and_load(self, simple_results, tmp_path):
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = Results.load(path)
        assert loaded.doc1_title == simple_results.doc1_title
        assert loaded.doc2_title == simple_results.doc2_title

    def test_matrices_close(self, simple_results, tmp_path):
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = Results.load(path)
        assert np.allclose(
            np.asarray(loaded.matrices["jaccard"]),
            simple_results.matrices["jaccard"],
            atol=1e-5,
        )

    def test_passages_preserved(self, simple_results, tmp_path):
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = Results.load(path)
        assert [p.text for p in loaded.passages1] == [p.text for p in simple_results.passages1]
        assert [p.label for p in loaded.passages1] == [p.label for p in simple_results.passages1]

    def test_lazy_top_k(self, simple_results, tmp_path):
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = Results.load(path)
        results = loaded.top_k(0, "jaccard", k=2)
        assert len(results) == 2

    def test_lazy_filter(self, simple_results, tmp_path):
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = Results.load(path)
        results = loaded.filter("jaccard", 0.0)
        assert len(results) > 0

    def test_hdf5_extension(self, simple_results, tmp_path):
        path = str(tmp_path / "r.hdf5")
        simple_results.save(path)
        loaded = Results.load(path)
        assert isinstance(loaded, Results)

    def test_pickle_after_hdf5_load(self, simple_results, tmp_path):
        """Pickling an HDF5-backed Results should produce a self-contained pickle."""
        h5_path = str(tmp_path / "r.h5")
        pkl_path = str(tmp_path / "r.pkl")
        simple_results.save(h5_path)
        loaded = Results.load(h5_path)
        loaded.save(pkl_path)
        loaded2 = Results.load(pkl_path)
        assert np.allclose(
            loaded2.matrices["jaccard"],
            simple_results.matrices["jaccard"],
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# ResultsCollection basics
# ---------------------------------------------------------------------------

class TestResultsCollection:
    def test_len(self, simple_collection):
        assert len(simple_collection) == 2

    def test_getitem(self, simple_collection):
        r = simple_collection[0, 0]
        assert isinstance(r, Results)

    def test_iter_keys(self, simple_collection):
        keys = list(simple_collection)
        assert (0, 0) in keys

    def test_items(self, simple_collection):
        for (i, j), r in simple_collection.items():
            assert isinstance(r, Results)

    def test_pairs_yields_titles(self, simple_collection):
        pairs = [(t1, t2) for t1, t2, _ in simple_collection.pairs()]
        assert len(pairs) == 2

    def test_repr(self, simple_collection):
        assert "2" in repr(simple_collection)


# ---------------------------------------------------------------------------
# ResultsCollection.filter
# ---------------------------------------------------------------------------

class TestResultsCollectionFilter:
    def test_returns_collection_matches(self, simple_collection):
        matches = simple_collection.filter("jaccard", 0.0)
        assert all(isinstance(m, CollectionMatch) for m in matches)

    def test_sorted_descending(self, simple_collection):
        matches = simple_collection.filter("jaccard", 0.0)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_has_doc_titles(self, simple_collection):
        matches = simple_collection.filter("jaccard", 0.0)
        for m in matches:
            assert m.doc1_title in ("Doc A", "Doc B")
            assert m.doc2_title in ("Doc A", "Doc B")

    def test_threshold_reduces_count(self, simple_collection):
        all_m = simple_collection.filter("jaccard", 0.0)
        high_m = simple_collection.filter("jaccard", 0.9)
        assert len(high_m) <= len(all_m)


# ---------------------------------------------------------------------------
# ResultsCollection — pickle backend
# ---------------------------------------------------------------------------

class TestResultsCollectionPickle:
    def test_save_and_load(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.pkl")
        simple_collection.save(path)
        loaded = ResultsCollection.load(path)
        assert len(loaded) == len(simple_collection)

    def test_matrices_preserved(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.pkl")
        simple_collection.save(path)
        loaded = ResultsCollection.load(path)
        orig = simple_collection[0, 0].matrices["jaccard"]
        loaded_mat = loaded[0, 0].matrices["jaccard"]
        assert np.allclose(orig, loaded_mat)


# ---------------------------------------------------------------------------
# ResultsCollection — SQLite backend
# ---------------------------------------------------------------------------

class TestResultsCollectionSQLite:
    def test_save_requires_threshold(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.db")
        with pytest.raises(ValueError, match="threshold"):
            simple_collection.save(path, backend="sqlite")

    def test_save_and_load(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.db")
        simple_collection.save(path, threshold=0.0)
        loaded = ResultsCollection.load(path)
        assert len(loaded) == len(simple_collection)

    def test_passages_preserved(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.db")
        simple_collection.save(path, threshold=0.0)
        loaded = ResultsCollection.load(path)
        orig_texts = [p.text for p in simple_collection[0, 0].passages1]
        loaded_texts = [p.text for p in loaded[0, 0].passages1]
        assert orig_texts == loaded_texts

    def test_filter_after_load(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.db")
        simple_collection.save(path, threshold=0.0)
        loaded = ResultsCollection.load(path)
        matches = loaded.filter("jaccard", 0.0)
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# ResultsCollection — HDF5 backend
# ---------------------------------------------------------------------------

class TestResultsCollectionHDF5:
    @pytest.fixture(autouse=True)
    def require_h5py(self):
        pytest.importorskip("h5py")

    def test_save_and_load(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.h5")
        simple_collection.save(path)
        loaded = ResultsCollection.load(path)
        assert len(loaded) == len(simple_collection)

    def test_matrices_close(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.h5")
        simple_collection.save(path)
        loaded = ResultsCollection.load(path)
        orig = simple_collection[0, 0].matrices["jaccard"]
        loaded_mat = np.asarray(loaded[0, 0].matrices["jaccard"])
        assert np.allclose(orig, loaded_mat, atol=1e-5)

    def test_filter_after_load(self, simple_collection, tmp_path):
        path = str(tmp_path / "col.h5")
        simple_collection.save(path)
        loaded = ResultsCollection.load(path)
        matches = loaded.filter("jaccard", 0.0)
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# crossref.load() top-level
# ---------------------------------------------------------------------------

class TestTopLevelLoad:
    def test_load_pickle(self, simple_results, tmp_path):
        import crossref
        path = str(tmp_path / "r.pkl")
        simple_results.save(path)
        loaded = crossref.load(path)
        assert isinstance(loaded, Results)

    def test_load_sqlite_results(self, simple_results, tmp_path):
        import crossref
        path = str(tmp_path / "r.db")
        simple_results.save(path, threshold=0.0)
        loaded = crossref.load(path)
        assert isinstance(loaded, Results)

    def test_load_sqlite_collection(self, simple_collection, tmp_path):
        import crossref
        path = str(tmp_path / "col.db")
        simple_collection.save(path, threshold=0.0)
        loaded = crossref.load(path)
        assert isinstance(loaded, ResultsCollection)

    def test_load_hdf5(self, simple_results, tmp_path):
        h5py = pytest.importorskip("h5py")
        import crossref
        path = str(tmp_path / "r.h5")
        simple_results.save(path)
        loaded = crossref.load(path)
        assert isinstance(loaded, Results)
