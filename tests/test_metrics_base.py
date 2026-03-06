"""Tests for crossref.metrics.base."""
import numpy as np
import pytest

from crossref.metrics.base import SimilarityMetric, _score_row, _score_self_row


# ---------------------------------------------------------------------------
# Module-level worker functions
# ---------------------------------------------------------------------------

class TestScoreRow:
    def test_applies_fn_to_each(self):
        fn = lambda a, b: len(a) + len(b)
        result = _score_row((fn, "hi", ["a", "bb", "ccc"]))
        assert result == [3, 4, 5]

    def test_empty_corpus(self):
        fn = lambda a, b: 0.0
        assert _score_row((fn, "x", [])) == []


class TestScoreSelfRow:
    def test_upper_triangle_only(self):
        fn = lambda a, b: 1.0
        result = _score_self_row((fn, 1, "text", ["a", "b", "c", "d"]))
        assert set(result.keys()) == {2, 3}
        assert 0 not in result and 1 not in result

    def test_values_correct(self):
        fn = lambda a, b: len(a) * len(b)
        corpus = ["a", "bb", "ccc"]
        result = _score_self_row((fn, 0, "a", corpus))
        # i=0: produces j=1 and j=2
        assert result[1] == len("a") * len("bb")
        assert result[2] == len("a") * len("ccc")


# ---------------------------------------------------------------------------
# SimilarityMetric ABC
# ---------------------------------------------------------------------------

class TestSimilarityMetricABC:
    def test_abstract_score_required(self):
        with pytest.raises(TypeError):
            SimilarityMetric()  # cannot instantiate abstract class

    def test_concrete_without_name_raises(self):
        with pytest.raises(TypeError, match="name"):
            class NoName(SimilarityMetric):
                def score(self, t1, t2):
                    return 0.0

    def test_abstract_intermediate_ok(self):
        # Abstract intermediate (still has abstract methods) is exempt from name check
        from crossref.metrics.syntactic import SyntacticSimilarityMetric
        # Should not raise — SyntacticSimilarityMetric has no name but is still abstract


class TestSimilarityMetricDefaults:
    """Test the default implementations on a minimal concrete subclass."""

    @pytest.fixture
    def minimal_metric(self):
        class ConstantMetric(SimilarityMetric):
            name = "constant"
            def score(self, t1, t2):
                return 0.5
        return ConstantMetric()

    def test_score_all_shape(self, minimal_metric):
        t1 = ["a", "b", "c"]
        t2 = ["x", "y"]
        result = minimal_metric.score_all(t1, t2)
        assert result.shape == (3, 2)

    def test_score_all_values(self, minimal_metric):
        result = minimal_metric.score_all(["a"], ["b"])
        assert result[0, 0] == pytest.approx(0.5)

    def test_score_self_shape(self, minimal_metric):
        result = minimal_metric.score_self(["a", "b", "c"])
        assert result.shape == (3, 3)

    def test_config_has_name(self, minimal_metric):
        assert minimal_metric.config() == {"name": "constant"}

    def test_explain_has_score(self, minimal_metric):
        ev = minimal_metric.explain("text1", "text2")
        assert "score" in ev
        assert ev["score"] == pytest.approx(0.5)
