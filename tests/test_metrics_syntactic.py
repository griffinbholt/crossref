"""Tests for crossref.metrics.syntactic."""
import numpy as np
import pytest

from crossref.metrics.syntactic import (
    NGramMetric,
    JaccardMetric,
    CharNGramMetric,
    LCSMetric,
    LevenshteinMetric,
    LCSubstringMetric,
    DiceMetric,
    ROUGELMetric,
    ROUGENMetric,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_score_all_shape(metric, texts1, texts2):
    m = metric.score_all(texts1, texts2)
    assert m.shape == (len(texts1), len(texts2))


def check_score_self_symmetry(metric, texts):
    m = metric.score_self(texts)
    assert m.shape == (len(texts), len(texts))
    assert np.allclose(m, m.T)


# ---------------------------------------------------------------------------
# JaccardMetric
# ---------------------------------------------------------------------------

class TestJaccardMetric:
    def test_identical(self):
        m = JaccardMetric()
        assert m.score("the quick fox", "the quick fox") == pytest.approx(1.0)

    def test_disjoint(self):
        m = JaccardMetric()
        assert m.score("aaa bbb ccc", "xxx yyy zzz") == pytest.approx(0.0)

    def test_partial_overlap(self):
        m = JaccardMetric()
        s = m.score("the quick fox", "the slow fox")
        assert 0.0 < s < 1.0

    def test_both_empty(self):
        m = JaccardMetric()
        assert m.score("", "") == pytest.approx(1.0)

    def test_bigram_shingles(self):
        m = JaccardMetric(n=2)
        s = m.score("the quick fox", "the quick fox")
        assert s == pytest.approx(1.0)

    def test_score_all_shape(self):
        m = JaccardMetric()
        check_score_all_shape(m, ["a b", "c d"], ["a b", "e f", "g h"])

    def test_score_self_symmetric(self):
        m = JaccardMetric()
        check_score_self_symmetry(m, ["a b c", "d e f", "a b d"])

    def test_score_self_diagonal_ones(self):
        m = JaccardMetric()
        texts = ["hello world", "foo bar"]
        mat = m.score_self(texts)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_explain_has_fields(self):
        m = JaccardMetric()
        ev = m.explain("the quick fox", "the slow fox")
        assert "score" in ev
        assert "shared" in ev
        assert "shared_count" in ev
        assert "union_count" in ev

    def test_config_roundtrip(self):
        m = JaccardMetric(n=2)
        assert m.config() == {"name": "jaccard", "n": 2}

    def test_score_in_range(self):
        m = JaccardMetric()
        s = m.score("hello world foo", "hello world bar")
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# CharNGramMetric
# ---------------------------------------------------------------------------

class TestCharNGramMetric:
    def test_identical(self):
        m = CharNGramMetric()
        assert m.score("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        m = CharNGramMetric()
        assert m.score("aaaa", "zzzz") == pytest.approx(0.0)

    def test_partial_overlap(self):
        m = CharNGramMetric()
        s = m.score("baptize", "baptise")
        assert s > 0.0

    def test_score_all_shape(self):
        m = CharNGramMetric()
        check_score_all_shape(m, ["hello", "world"], ["hello", "other"])

    def test_score_self_symmetric(self):
        m = CharNGramMetric()
        check_score_self_symmetry(m, ["hello", "world", "test"])

    def test_config(self):
        m = CharNGramMetric(n=3)
        assert m.config() == {"name": "charngram", "n": 3}


# ---------------------------------------------------------------------------
# LCSMetric
# ---------------------------------------------------------------------------

class TestLCSMetric:
    def test_identical(self):
        m = LCSMetric()
        assert m.score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_disjoint(self):
        m = LCSMetric()
        assert m.score("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)

    def test_subsequence(self):
        # "the fox" is a subsequence of "the quick brown fox"
        m = LCSMetric()
        s = m.score("the fox", "the quick brown fox")
        assert s > 0.0

    def test_both_empty(self):
        m = LCSMetric()
        assert m.score("", "") == pytest.approx(0.0)

    def test_score_all_shape(self):
        m = LCSMetric()
        check_score_all_shape(m, ["a b c", "d e"], ["a b", "f g"])

    def test_score_self_symmetric(self):
        m = LCSMetric()
        check_score_self_symmetry(m, ["hello world", "foo bar", "hello foo"])

    def test_explain_has_subsequence(self):
        m = LCSMetric()
        ev = m.explain("the quick fox", "the slow fox")
        assert "score" in ev
        assert "common_subsequence" in ev
        assert isinstance(ev["common_subsequence"], list)

    def test_config(self):
        m = LCSMetric()
        assert m.config()["name"] == "lcs"


# ---------------------------------------------------------------------------
# LevenshteinMetric
# ---------------------------------------------------------------------------

class TestLevenshteinMetric:
    def test_identical(self):
        m = LevenshteinMetric()
        assert m.score("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        m = LevenshteinMetric()
        s = m.score("aaa bbb ccc", "xxx yyy zzz")
        assert s < 1.0

    def test_one_word_different(self):
        m = LevenshteinMetric()
        s = m.score("the quick fox", "the slow fox")
        assert 0.0 < s < 1.0

    def test_both_empty(self):
        m = LevenshteinMetric()
        assert m.score("", "") == pytest.approx(1.0)

    def test_score_all_shape(self):
        m = LevenshteinMetric()
        check_score_all_shape(m, ["a b", "c d"], ["a b", "c d"])

    def test_score_self_symmetric(self):
        m = LevenshteinMetric()
        check_score_self_symmetry(m, ["hello world", "foo bar"])

    def test_explain_has_fields(self):
        m = LevenshteinMetric()
        ev = m.explain("hello world", "hello earth")
        assert "score" in ev
        assert "edit_distance" in ev
        assert "len1" in ev
        assert "len2" in ev

    def test_score_in_range(self):
        m = LevenshteinMetric()
        assert 0.0 <= m.score("abc def", "abc xyz") <= 1.0


# ---------------------------------------------------------------------------
# LCSubstringMetric
# ---------------------------------------------------------------------------

class TestLCSubstringMetric:
    def test_identical(self):
        m = LCSubstringMetric()
        assert m.score("the quick fox", "the quick fox") == pytest.approx(1.0)

    def test_no_match(self):
        m = LCSubstringMetric()
        assert m.score("aaa bbb", "xxx yyy") == pytest.approx(0.0)

    def test_partial_match(self):
        m = LCSubstringMetric()
        s = m.score("the quick brown fox", "the quick cat")
        assert 0.0 < s < 1.0

    def test_score_all_shape(self):
        m = LCSubstringMetric()
        check_score_all_shape(m, ["a b c", "d e f"], ["a b", "d e"])

    def test_score_self_symmetric(self):
        m = LCSubstringMetric()
        check_score_self_symmetry(m, ["hello world", "world peace"])

    def test_explain_has_substring(self):
        m = LCSubstringMetric()
        ev = m.explain("the quick brown fox", "the quick cat jumped")
        assert "common_substring" in ev
        assert isinstance(ev["common_substring"], list)


# ---------------------------------------------------------------------------
# DiceMetric
# ---------------------------------------------------------------------------

class TestDiceMetric:
    def test_identical(self):
        m = DiceMetric()
        assert m.score("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        m = DiceMetric()
        assert m.score("aaa bbb", "xxx yyy") == pytest.approx(0.0)

    def test_partial(self):
        m = DiceMetric()
        s = m.score("hello world", "hello earth")
        assert 0.0 < s < 1.0

    def test_score_all_shape(self):
        m = DiceMetric()
        check_score_all_shape(m, ["a b", "c d"], ["a b", "e f"])

    def test_score_self_symmetric(self):
        m = DiceMetric()
        check_score_self_symmetry(m, ["hello world", "foo bar"])

    def test_config(self):
        m = DiceMetric(n=2)
        assert m.config() == {"name": "dice", "n": 2}


# ---------------------------------------------------------------------------
# ROUGELMetric
# ---------------------------------------------------------------------------

class TestROUGELMetric:
    def test_identical(self):
        m = ROUGELMetric()
        # ROUGE-L of identical texts = 2n/(2n) = 1
        s = m.score("the quick brown fox", "the quick brown fox")
        assert s == pytest.approx(1.0)

    def test_no_overlap(self):
        m = ROUGELMetric()
        s = m.score("aaa bbb", "xxx yyy")
        assert s == pytest.approx(0.0)

    def test_partial(self):
        m = ROUGELMetric()
        s = m.score("the quick fox", "the slow fox")
        assert 0.0 < s < 1.0

    def test_score_all_shape(self):
        m = ROUGELMetric()
        check_score_all_shape(m, ["a b c", "d e"], ["a b", "c d e"])

    def test_score_self_symmetric(self):
        m = ROUGELMetric()
        check_score_self_symmetry(m, ["hello world", "foo bar"])

    def test_config(self):
        m = ROUGELMetric()
        assert m.config()["name"] == "rouge_l"


# ---------------------------------------------------------------------------
# ROUGENMetric
# ---------------------------------------------------------------------------

class TestROUGENMetric:
    def test_identical(self):
        m = ROUGENMetric()
        s = m.score("hello world", "hello world")
        assert s == pytest.approx(1.0)

    def test_disjoint(self):
        m = ROUGENMetric()
        s = m.score("aaa bbb", "xxx yyy")
        assert s == pytest.approx(0.0)

    def test_partial(self):
        m = ROUGENMetric()
        s = m.score("hello world foo", "hello world bar")
        assert 0.0 < s < 1.0

    def test_bigram(self):
        m = ROUGENMetric(n=2)
        s = m.score("the quick fox", "the quick fox")
        assert s == pytest.approx(1.0)

    def test_score_all_shape(self):
        m = ROUGENMetric()
        check_score_all_shape(m, ["a b", "c d"], ["a b", "e f"])

    def test_score_self_symmetric(self):
        m = ROUGENMetric()
        check_score_self_symmetry(m, ["hello world", "foo bar"])

    def test_config(self):
        m = ROUGENMetric(n=2)
        assert m.config() == {"name": "rouge_n", "n": 2}


# ---------------------------------------------------------------------------
# NGramMetric
# ---------------------------------------------------------------------------

class TestNGramMetric:
    def test_identical_score_one(self):
        m = NGramMetric()
        s = m.score("the quick brown fox", "the quick brown fox")
        assert s == pytest.approx(1.0)

    def test_disjoint_score_zero(self):
        m = NGramMetric()
        s = m.score("alpha beta gamma delta", "zeta eta theta iota")
        assert s == pytest.approx(0.0)

    def test_partial_overlap(self):
        m = NGramMetric()
        s = m.score("the quick brown fox", "the quick brown cat")
        assert 0.0 < s < 1.0

    def test_empty_texts(self):
        m = NGramMetric()
        s = m.score("", "")
        assert s == pytest.approx(0.0)

    def test_score_all_shape(self):
        m = NGramMetric()
        mat = m.score_all(["a b c d", "e f g h"], ["a b c d", "x y z w"])
        assert mat.shape == (2, 2)

    def test_score_all_diagonal_high(self):
        m = NGramMetric()
        texts = ["the quick brown fox jumps", "the lord said unto them"]
        mat = m.score_all(texts, texts)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_score_self_symmetric(self):
        m = NGramMetric()
        texts = ["the quick brown fox", "jumps over the lazy dog"]
        mat = m.score_self(texts)
        assert mat.shape == (2, 2)
        assert np.allclose(mat, mat.T)

    def test_score_self_diagonal_ones(self):
        m = NGramMetric()
        texts = ["the quick brown fox jumps", "hello world foo bar"]
        mat = m.score_self(texts)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_scores_in_range(self):
        m = NGramMetric()
        mat = m.score_all(
            ["the lord is my shepherd", "in the beginning"],
            ["the lord is my shepherd", "something different"],
        )
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0 + 1e-6)

    def test_explain_returns_common_ngrams(self):
        m = NGramMetric()
        ev = m.explain("the quick brown fox", "the quick brown fox")
        assert "score" in ev
        assert "common_ngrams" in ev
        assert isinstance(ev["common_ngrams"], dict)

    def test_config_roundtrip(self):
        m = NGramMetric(n_min=3, n_max=5)
        cfg = m.config()
        assert cfg["name"] == "ngram"
        assert cfg["n_min"] == 3
        assert cfg["n_max"] == 5

    def test_custom_stopphrases(self):
        m1 = NGramMetric(custom_stopphrases={"the lord"})
        m2 = NGramMetric()
        text = "the lord commanded Moses speak these words Israel"
        s1 = m1.score(text, text)
        s2 = m2.score(text, text)
        # Both should be 1.0 on identical texts
        assert s1 == pytest.approx(1.0)
        assert s2 == pytest.approx(1.0)
        # Scores should be in valid range with custom stopphrases
        s_partial = m1.score("the lord commanded Moses speak Israel", "Moses speak Israel commanded")
        assert 0.0 <= s_partial <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# NCDMetric (no external deps)
# ---------------------------------------------------------------------------

class TestNCDMetric:
    def test_identical_high_score(self):
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric()
        s = m.score("hello world " * 20, "hello world " * 20)
        assert s > 0.5

    def test_disjoint_lower_score(self):
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric()
        s_same = m.score("aaa " * 30, "aaa " * 30)
        s_diff = m.score("aaa " * 30, "zzz " * 30)
        assert s_same > s_diff

    def test_score_in_range(self):
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric()
        s = m.score("hello world", "hello earth")
        assert -0.1 <= s <= 1.1  # NCD can go slightly outside [0,1] theoretically

    def test_score_all_shape(self):
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric()
        mat = m.score_all(["a " * 20, "b " * 20], ["a " * 20, "c " * 20])
        assert mat.shape == (2, 2)

    def test_config(self):
        from crossref.metrics.statistical import NCDMetric
        m = NCDMetric(level=5)
        assert m.config()["name"] == "ncd"
        assert m.config()["level"] == 5
