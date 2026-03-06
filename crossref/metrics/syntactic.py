import logging

import numpy as np

from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from .base import SimilarityMetric
from ..preprocessing import remove_punctuation, normalize_text, remove_stopwords

logger = logging.getLogger(__name__)


class SyntacticSimilarityMetric(SimilarityMetric):
    pass


class NGramMetric(SyntacticSimilarityMetric):
    name = "ngram"

    def __init__(
        self,
        n_min: int = 2,
        n_max: int | None = None,
        custom_stopphrases: set[str] | None = None,
        workers: int = 1,
    ):
        self.n_min: int = n_min
        self.n_max: int | None = n_max
        self.custom_stopphrases: set[str] = set(custom_stopphrases) if custom_stopphrases else set()
        self.workers: int = workers

    def score(self, text1: str, text2: str, return_full: bool = False) -> float | tuple[float, dict[int, set[tuple[str, ...]]]]:
        ngrams_text1: dict[int, set[tuple[str, ...]]] = self._generate_ngrams(text1)
        ngrams_text2: dict[int, set[tuple[str, ...]]] = self._generate_ngrams(text2)
        common_ngrams: dict[int, set[tuple[str, ...]]] = self._find_common_largest_ngrams(ngrams_text1, ngrams_text2)
        raw: float = self._compute_ngram_score(common_ngrams)
        norm = min(self._self_score(ngrams_text1), self._self_score(ngrams_text2))
        score = raw / norm if norm > 0 else 0.0
        if return_full:
            return score, common_ngrams
        return score

    def score_all(
        self,
        texts1: list[str],
        texts2: list[str],
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[list[dict[int, set[tuple[str, ...]]]]]]:
        logger.debug("NGramMetric.score_all: %dx%d", len(texts1), len(texts2))
        ngram_cache1: list[dict[int, set[tuple[str, ...]]]] = [self._generate_ngrams(text) for text in texts1]
        ngram_cache2: list[dict[int, set[tuple[str, ...]]]] = [self._generate_ngrams(text) for text in texts2]
        self_scores1 = [self._self_score(ng) for ng in ngram_cache1]
        self_scores2 = [self._self_score(ng) for ng in ngram_cache2]

        if self.workers > 1 and not return_full:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_row
            tasks = [(self._pair_score, ng1, ngram_cache2) for ng1 in ngram_cache1]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                scores = np.array(list(ex.map(_score_row, tasks)))
        else:
            complete_common_ngrams: list[list[dict[int, set[tuple[str, ...]]]]] = [[None] * len(texts2) for _ in range(len(texts1))]
            scores = np.zeros((len(texts1), len(texts2)))
            for i, ng1 in enumerate(ngram_cache1):
                for j, ng2 in enumerate(ngram_cache2):
                    common_ngrams = self._find_common_largest_ngrams(ng1, ng2)
                    scores[i, j] = self._compute_ngram_score(common_ngrams)
                    if return_full:
                        complete_common_ngrams[i][j] = common_ngrams

        norm = np.minimum(np.array(self_scores1)[:, np.newaxis], np.array(self_scores2)[np.newaxis, :])
        norm = np.where(norm == 0, 1.0, norm)  # avoid division by zero for empty passages
        nmlzd_scores = scores / norm

        if return_full:
            return nmlzd_scores, complete_common_ngrams
        return nmlzd_scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        logger.debug("NGramMetric.score_self: %d passages", len(texts))
        ngram_cache = [self._generate_ngrams(text) for text in texts]
        self_scores = [self._self_score(ng) for ng in ngram_cache]
        n = len(texts)
        scores = np.zeros((n, n))

        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_self_row
            tasks = [(self._pair_score, i, ngram_cache[i], ngram_cache) for i in range(n)]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                for i, row_dict in enumerate(ex.map(_score_self_row, tasks)):
                    for j, s in row_dict.items():
                        scores[i, j] = scores[j, i] = s
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    common = self._find_common_largest_ngrams(ngram_cache[i], ngram_cache[j])
                    scores[i, j] = scores[j, i] = self._compute_ngram_score(common)

        for i in range(n):
            scores[i, i] = self_scores[i]
        norm = np.minimum(np.array(self_scores)[:, np.newaxis], np.array(self_scores)[np.newaxis, :])
        norm = np.where(norm == 0, 1.0, norm)
        return scores / norm

    def _pair_score(self, ng1: dict, ng2: dict) -> float:
        """Unnormalized score for one pair; used by ProcessPool workers."""
        return self._compute_ngram_score(self._find_common_largest_ngrams(ng1, ng2))

    def _self_score(self, text_ngrams: dict[int, set[tuple[str, ...]]]) -> float:
        """Score of a text compared against itself (used as normalization denominator).

        Computes the actual self-comparison score so that normalization is exact
        even when n_max is finite and there are multiple non-overlapping top-level n-grams.
        """
        if not text_ngrams:
            return 0.0
        common = self._find_common_largest_ngrams(text_ngrams, text_ngrams)
        return self._compute_ngram_score(common)

    def _generate_ngrams(self, text: str) -> dict[int, set[tuple[str, ...]]]:
        """Generate n-grams for all n in [n_min, n_max] after preprocessing."""
        cleaned_text: str = remove_stopwords(normalize_text(remove_punctuation(text)), self.custom_stopphrases)
        tokens: list[str] = word_tokenize(cleaned_text)
        if len(tokens) < self.n_min:
            return {}
        n_max = self.n_max if self.n_max is not None else len(tokens)
        return {n: set(ngrams(tokens, n)) for n in range(self.n_min, min(n_max, len(tokens)) + 1)}

    def _find_common_largest_ngrams(
            self,
            ngrams_text1: dict[int, set[tuple[str, ...]]],
            ngrams_text2: dict[int, set[tuple[str, ...]]]
        ) -> dict[int, set[tuple[str, ...]]]:
        """Find common n-grams, keeping only the largest non-overlapping ones."""
        if not ngrams_text1 or not ngrams_text2:
            return {}

        n_max: int = min(max(ngrams_text1.keys()), max(ngrams_text2.keys()))
        common_ngrams = {n: ngrams_text1[n] & ngrams_text2[n] for n in range(self.n_min, n_max + 1)}

        filtered_ngrams: dict[int, set[tuple[str, ...]]] = {}
        for n in range(n_max, self.n_min - 1, -1):
            for ngram in common_ngrams[n]:
                if n not in filtered_ngrams:
                    filtered_ngrams[n] = set()
                filtered_ngrams[n].add(ngram)
                if n > self.n_min:
                    for sub_n in range(n - 1, self.n_min - 1, -1):
                        for i in range(n - sub_n + 1):
                            sub_ngram = ngram[i:i+sub_n]
                            if sub_ngram in common_ngrams[sub_n]:
                                common_ngrams[sub_n].remove(sub_ngram)

        return filtered_ngrams

    def _compute_ngram_score(self, common_ngrams: dict[int, set[tuple[str, ...]]]) -> float:
        return float(sum(n * len(phrases) for n, phrases in common_ngrams.items()))

    def config(self) -> dict:
        return {
            "name": self.name,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "custom_stopphrases": sorted(self.custom_stopphrases),
            "workers": self.workers,
        }

    def explain(self, text1: str, text2: str) -> dict:
        """Return normalized score and common n-grams between text1 and text2.

        The score matches the value stored in the similarity matrix.

        Returns:
            {
                "score": float,  # normalized, matches Results.matrices[metric][i, j]
                "common_ngrams": {n: {(word, ...), ...}, ...}
                    Only the largest non-overlapping n-grams are included.
            }
        """
        ng1 = self._generate_ngrams(text1)
        ng2 = self._generate_ngrams(text2)
        common = self._find_common_largest_ngrams(ng1, ng2)
        raw_score = self._compute_ngram_score(common)
        norm = min(self._self_score(ng1), self._self_score(ng2))
        normalized_score = raw_score / norm if norm > 0 else 0.0
        return {"score": normalized_score, "common_ngrams": common}


class JaccardMetric(SyntacticSimilarityMetric):
    """Jaccard similarity on word (or word n-gram) sets.

    |A ∩ B| / |A ∪ B|. Unordered, unweighted word overlap — the simplest
    interpretable syntactic baseline. Parameterize n > 1 for n-gram shingles
    instead of unigrams (captures some word-order sensitivity).
    """
    name = "jaccard"

    def __init__(self, n: int = 1):
        """
        Args:
            n: Shingle size. n=1 (default) = word unigrams; n=2 = bigrams, etc.
        """
        self.n = n

    def score(self, text1: str, text2: str) -> float:
        s1, s2 = self._shingles(text1), self._shingles(text2)
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        shingles1 = [self._shingles(t) for t in texts1]
        shingles2 = [self._shingles(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, s1 in enumerate(shingles1):
            for j, s2 in enumerate(shingles2):
                union = len(s1 | s2)
                scores[i, j] = len(s1 & s2) / union if union > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        shingles = [self._shingles(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                union = len(shingles[i] | shingles[j])
                scores[i, j] = scores[j, i] = len(shingles[i] & shingles[j]) / union if union > 0 else 1.0
        return scores

    def explain(self, text1: str, text2: str) -> dict:
        s1, s2 = self._shingles(text1), self._shingles(text2)
        intersection = s1 & s2
        union = s1 | s2
        shared = sorted(x if isinstance(x, str) else " ".join(x) for x in intersection)
        return {
            "score": len(intersection) / len(union) if union else 1.0,
            "shared": shared,
            "shared_count": len(intersection),
            "union_count": len(union),
        }

    def _shingles(self, text: str) -> set:
        tokens = text.lower().split()
        if self.n == 1:
            return set(tokens)
        return {tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)}

    def config(self) -> dict:
        return {"name": self.name, "n": self.n}


class CharNGramMetric(SyntacticSimilarityMetric):
    """Character n-gram Jaccard similarity.

    Jaccard similarity on sets of character n-grams. More robust to morphological
    variation and archaic spelling differences than word-level metrics:
    "baptize"/"baptise", "saith"/"said", etc. still share most character n-grams.
    Useful for comparing passages across translations.
    """
    name = "charngram"

    def __init__(self, n: int = 4):
        """
        Args:
            n: Character n-gram size. Default 4 (4-grams). Smaller = more
               tolerant of variation; larger = more precise matching.
        """
        self.n = n

    def score(self, text1: str, text2: str) -> float:
        s1, s2 = self._ngrams(text1), self._ngrams(text2)
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        ngrams1 = [self._ngrams(t) for t in texts1]
        ngrams2 = [self._ngrams(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, s1 in enumerate(ngrams1):
            for j, s2 in enumerate(ngrams2):
                union = len(s1 | s2)
                scores[i, j] = len(s1 & s2) / union if union > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        ngrams = [self._ngrams(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                union = len(ngrams[i] | ngrams[j])
                scores[i, j] = scores[j, i] = len(ngrams[i] & ngrams[j]) / union if union > 0 else 1.0
        return scores

    def _ngrams(self, text: str) -> set:
        text = text.lower()
        return {text[i:i + self.n] for i in range(len(text) - self.n + 1)} if len(text) >= self.n else set()

    def config(self) -> dict:
        return {"name": self.name, "n": self.n}


class LCSMetric(SyntacticSimilarityMetric):
    """Longest Common Subsequence similarity (word-level).

    Finds the longest sequence of words that appear in both passages in the
    same order, but not necessarily adjacent. Normalized by min(len1, len2)
    so that 1.0 = one passage is a subsequence of the other.

    Different from NGramMetric: "And it came to [other words] pass" partially
    matches "And it came to pass". Useful for detecting passages where one text
    expands another by inserting elaborating material in the middle of a quote.

    No dependencies — classic O(n·m) DP algorithm.
    """
    name = "lcs"

    def __init__(self, workers: int = 1):
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        t1, t2 = text1.lower().split(), text2.lower().split()
        length = self._lcs_length(t1, t2)
        denom = min(len(t1), len(t2))
        return length / denom if denom > 0 else 0.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        tokens1 = [t.lower().split() for t in texts1]
        tokens2 = [t.lower().split() for t in texts2]
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_row
            tasks = [(self._pair_score, t1, tokens2) for t1 in tokens1]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                return np.array(list(ex.map(_score_row, tasks)))
        scores = np.zeros((len(texts1), len(texts2)))
        for i, t1 in enumerate(tokens1):
            for j, t2 in enumerate(tokens2):
                denom = min(len(t1), len(t2))
                scores[i, j] = self._lcs_length(t1, t2) / denom if denom > 0 else 0.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        tokens = [t.lower().split() for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_self_row
            tasks = [(self._pair_score, i, tokens[i], tokens) for i in range(n)]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                for i, row_dict in enumerate(ex.map(_score_self_row, tasks)):
                    for j, s in row_dict.items():
                        scores[i, j] = scores[j, i] = s
            return scores
        for i in range(n):
            for j in range(i + 1, n):
                denom = min(len(tokens[i]), len(tokens[j]))
                scores[i, j] = scores[j, i] = self._lcs_length(tokens[i], tokens[j]) / denom if denom > 0 else 0.0
        return scores

    def _pair_score(self, t1: list[str], t2: list[str]) -> float:
        denom = min(len(t1), len(t2))
        return self._lcs_length(t1, t2) / denom if denom > 0 else 0.0

    def explain(self, text1: str, text2: str) -> dict:
        t1, t2 = text1.lower().split(), text2.lower().split()
        sequence = self._lcs_sequence(t1, t2)
        denom = min(len(t1), len(t2))
        return {
            "score": len(sequence) / denom if denom > 0 else 0.0,
            "common_subsequence": sequence,
        }

    def _lcs_length(self, tokens1: list[str], tokens2: list[str]) -> int:
        prev = [0] * (len(tokens2) + 1)
        curr = [0] * (len(tokens2) + 1)
        for token1 in tokens1:
            for j, token2 in enumerate(tokens2, 1):
                curr[j] = prev[j - 1] + 1 if token1 == token2 else max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (len(tokens2) + 1)
        return prev[len(tokens2)]

    def _lcs_sequence(self, tokens1: list[str], tokens2: list[str]) -> list[str]:
        """Recover the actual LCS word sequence via full DP table backtracking."""
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        result, i, j = [], m, n
        while i > 0 and j > 0:
            if tokens1[i - 1] == tokens2[j - 1]:
                result.append(tokens1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return result[::-1]

    def config(self) -> dict:
        return {"name": self.name, "workers": self.workers}


class LevenshteinMetric(SyntacticSimilarityMetric):
    """Word-level Levenshtein (edit distance) similarity.

    Minimum word insertions, deletions, and substitutions to transform one
    passage into another. Normalized by max(len1, len2) → [0, 1] where
    1.0 = identical and 0.0 = no words in common.

    Captures near-identical passages with minor word-level edits — quotations
    that change a few words, scribal variants, or passages that differ by one
    clause. No dependencies — classic O(n·m) DP algorithm.
    """
    name = "levenshtein"

    def __init__(self, workers: int = 1):
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        t1, t2 = text1.lower().split(), text2.lower().split()
        dist = self._edit_distance(t1, t2)
        denom = max(len(t1), len(t2))
        return 1.0 - dist / denom if denom > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        tokens1 = [t.lower().split() for t in texts1]
        tokens2 = [t.lower().split() for t in texts2]
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_row
            tasks = [(self._pair_score, t1, tokens2) for t1 in tokens1]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                return np.array(list(ex.map(_score_row, tasks)))
        scores = np.zeros((len(texts1), len(texts2)))
        for i, t1 in enumerate(tokens1):
            for j, t2 in enumerate(tokens2):
                dist = self._edit_distance(t1, t2)
                denom = max(len(t1), len(t2))
                scores[i, j] = 1.0 - dist / denom if denom > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        tokens = [t.lower().split() for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_self_row
            tasks = [(self._pair_score, i, tokens[i], tokens) for i in range(n)]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                for i, row_dict in enumerate(ex.map(_score_self_row, tasks)):
                    for j, s in row_dict.items():
                        scores[i, j] = scores[j, i] = s
            return scores
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._edit_distance(tokens[i], tokens[j])
                denom = max(len(tokens[i]), len(tokens[j]))
                scores[i, j] = scores[j, i] = 1.0 - dist / denom if denom > 0 else 1.0
        return scores

    def _pair_score(self, t1: list[str], t2: list[str]) -> float:
        dist = self._edit_distance(t1, t2)
        denom = max(len(t1), len(t2))
        return 1.0 - dist / denom if denom > 0 else 1.0

    def explain(self, text1: str, text2: str) -> dict:
        t1, t2 = text1.lower().split(), text2.lower().split()
        dist = self._edit_distance(t1, t2)
        denom = max(len(t1), len(t2))
        return {
            "score": 1.0 - dist / denom if denom > 0 else 1.0,
            "edit_distance": dist,
            "len1": len(t1),
            "len2": len(t2),
        }

    def _edit_distance(self, tokens1: list[str], tokens2: list[str]) -> int:
        prev = list(range(len(tokens2) + 1))
        curr = [0] * (len(tokens2) + 1)
        for i, token1 in enumerate(tokens1, 1):
            curr[0] = i
            for j, token2 in enumerate(tokens2, 1):
                curr[j] = prev[j - 1] if token1 == token2 else 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, [0] * (len(tokens2) + 1)
        return prev[len(tokens2)]

    def config(self) -> dict:
        return {"name": self.name, "workers": self.workers}


class LCSubstringMetric(SyntacticSimilarityMetric):
    """Longest Common Substring similarity (word-level, contiguous).

    Stricter than LCSMetric (subsequence): matched words must be adjacent.
    Score = longest_common_substring_length / min(len1, len2).

    Useful for detecting direct verbatim quotations where no words are inserted
    in the middle of the match — complements LCSMetric, which allows gaps.

    No dependencies — O(n·m) DP algorithm.
    """
    name = "lcsubstring"

    def __init__(self, workers: int = 1):
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        t1, t2 = text1.lower().split(), text2.lower().split()
        denom = min(len(t1), len(t2))
        return self._lcs_length(t1, t2) / denom if denom > 0 else 0.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        tokens1 = [t.lower().split() for t in texts1]
        tokens2 = [t.lower().split() for t in texts2]
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_row
            tasks = [(self._pair_score, t1, tokens2) for t1 in tokens1]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                return np.array(list(ex.map(_score_row, tasks)))
        scores = np.zeros((len(texts1), len(texts2)))
        for i, t1 in enumerate(tokens1):
            for j, t2 in enumerate(tokens2):
                denom = min(len(t1), len(t2))
                scores[i, j] = self._lcs_length(t1, t2) / denom if denom > 0 else 0.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        tokens = [t.lower().split() for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_self_row
            tasks = [(self._pair_score, i, tokens[i], tokens) for i in range(n)]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                for i, row_dict in enumerate(ex.map(_score_self_row, tasks)):
                    for j, s in row_dict.items():
                        scores[i, j] = scores[j, i] = s
            return scores
        for i in range(n):
            for j in range(i + 1, n):
                denom = min(len(tokens[i]), len(tokens[j]))
                scores[i, j] = scores[j, i] = self._lcs_length(tokens[i], tokens[j]) / denom if denom > 0 else 0.0
        return scores

    def _pair_score(self, t1: list[str], t2: list[str]) -> float:
        denom = min(len(t1), len(t2))
        return self._lcs_length(t1, t2) / denom if denom > 0 else 0.0

    def explain(self, text1: str, text2: str) -> dict:
        t1, t2 = text1.lower().split(), text2.lower().split()
        length, start = self._lcs_with_start(t1, t2)
        denom = min(len(t1), len(t2))
        return {
            "score": length / denom if denom > 0 else 0.0,
            "common_substring": t1[start:start + length] if length > 0 else [],
        }

    def _lcs_length(self, tokens1: list[str], tokens2: list[str]) -> int:
        prev = [0] * (len(tokens2) + 1)
        curr = [0] * (len(tokens2) + 1)
        best = 0
        for t1 in tokens1:
            for j, t2 in enumerate(tokens2, 1):
                curr[j] = prev[j - 1] + 1 if t1 == t2 else 0
                if curr[j] > best:
                    best = curr[j]
            prev, curr = curr, [0] * (len(tokens2) + 1)
        return best

    def _lcs_with_start(self, tokens1: list[str], tokens2: list[str]) -> tuple[int, int]:
        """Returns (length, start_index_in_tokens1)."""
        m, n = len(tokens1), len(tokens2)
        best_len, best_end = 0, 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > best_len:
                        best_len = dp[i][j]
                        best_end = i
        return best_len, best_end - best_len

    def config(self) -> dict:
        return {"name": self.name, "workers": self.workers}


class DiceMetric(SyntacticSimilarityMetric):
    """Dice coefficient (2|A∩B| / (|A|+|B|)) on word or n-gram sets.

    Monotone variant of Jaccard: Dice = 2J / (1+J). Gives more weight to
    shared terms relative to individual passage sizes — slightly less
    conservative than Jaccard at mid-range overlap.

    Parameterize n > 1 for n-gram shingles (some word-order sensitivity).
    """
    name = "dice"

    def __init__(self, n: int = 1):
        self.n = n

    def score(self, text1: str, text2: str) -> float:
        s1, s2 = self._shingles(text1), self._shingles(text2)
        denom = len(s1) + len(s2)
        return 2 * len(s1 & s2) / denom if denom > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        sh1 = [self._shingles(t) for t in texts1]
        sh2 = [self._shingles(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, s1 in enumerate(sh1):
            for j, s2 in enumerate(sh2):
                denom = len(s1) + len(s2)
                scores[i, j] = 2 * len(s1 & s2) / denom if denom > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        shingles = [self._shingles(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                denom = len(shingles[i]) + len(shingles[j])
                scores[i, j] = scores[j, i] = 2 * len(shingles[i] & shingles[j]) / denom if denom > 0 else 1.0
        return scores

    def _shingles(self, text: str) -> set:
        tokens = text.lower().split()
        if self.n == 1:
            return set(tokens)
        return {tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)}

    def config(self) -> dict:
        return {"name": self.name, "n": self.n}


class ROUGELMetric(SyntacticSimilarityMetric):
    """ROUGE-L F1 score based on Longest Common Subsequence.

    F1 = 2 * LCS(t1,t2) / (len(t1) + len(t2)).

    Unlike LCSMetric which normalizes by min(len1, len2), ROUGE-L uses the
    harmonic mean of precision and recall — penalizing more when one passage
    is much longer than the other. Symmetric by construction.
    """
    name = "rouge_l"

    def __init__(self, workers: int = 1):
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        t1, t2 = text1.lower().split(), text2.lower().split()
        denom = len(t1) + len(t2)
        return 2 * self._lcs_length(t1, t2) / denom if denom > 0 else 0.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        tokens1 = [t.lower().split() for t in texts1]
        tokens2 = [t.lower().split() for t in texts2]
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_row
            tasks = [(self._pair_score, t1, tokens2) for t1 in tokens1]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                return np.array(list(ex.map(_score_row, tasks)))
        scores = np.zeros((len(texts1), len(texts2)))
        for i, t1 in enumerate(tokens1):
            for j, t2 in enumerate(tokens2):
                denom = len(t1) + len(t2)
                scores[i, j] = 2 * self._lcs_length(t1, t2) / denom if denom > 0 else 0.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        tokens = [t.lower().split() for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        if self.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from .base import _score_self_row
            tasks = [(self._pair_score, i, tokens[i], tokens) for i in range(n)]
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                for i, row_dict in enumerate(ex.map(_score_self_row, tasks)):
                    for j, s in row_dict.items():
                        scores[i, j] = scores[j, i] = s
            return scores
        for i in range(n):
            for j in range(i + 1, n):
                denom = len(tokens[i]) + len(tokens[j])
                scores[i, j] = scores[j, i] = 2 * self._lcs_length(tokens[i], tokens[j]) / denom if denom > 0 else 0.0
        return scores

    def _pair_score(self, t1: list[str], t2: list[str]) -> float:
        denom = len(t1) + len(t2)
        return 2 * self._lcs_length(t1, t2) / denom if denom > 0 else 0.0

    def _lcs_length(self, tokens1: list[str], tokens2: list[str]) -> int:
        prev = [0] * (len(tokens2) + 1)
        curr = [0] * (len(tokens2) + 1)
        for token1 in tokens1:
            for j, token2 in enumerate(tokens2, 1):
                curr[j] = prev[j - 1] + 1 if token1 == token2 else max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (len(tokens2) + 1)
        return prev[len(tokens2)]

    def config(self) -> dict:
        return {"name": self.name, "workers": self.workers}


class ROUGENMetric(SyntacticSimilarityMetric):
    """ROUGE-N F1 score based on n-gram count overlap.

    F1 = 2 * matched / (ngrams_in_t1 + ngrams_in_t2) using multiset matching:
    each n-gram can match multiple times proportional to its frequency in both
    texts. Unlike Jaccard/CharNGram (set-based), ROUGE-N rewards repetition.

    n=1 (ROUGE-1): word coverage. n=2 (ROUGE-2): phrase-level fluency.
    Symmetric by construction.
    """
    name = "rouge_n"

    def __init__(self, n: int = 1):
        self.n = n

    def score(self, text1: str, text2: str) -> float:
        return self._rouge_n(text1.lower().split(), text2.lower().split())

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        tokens1 = [t.lower().split() for t in texts1]
        tokens2 = [t.lower().split() for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, t1 in enumerate(tokens1):
            for j, t2 in enumerate(tokens2):
                scores[i, j] = self._rouge_n(t1, t2)
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        tokens = [t.lower().split() for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                scores[i, j] = scores[j, i] = self._rouge_n(tokens[i], tokens[j])
        return scores

    def _ngram_counts(self, tokens: list[str]):
        from collections import Counter
        return Counter(tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1))

    def _rouge_n(self, tokens1: list[str], tokens2: list[str]) -> float:
        c1, c2 = self._ngram_counts(tokens1), self._ngram_counts(tokens2)
        matched = sum(min(c1[ng], c2[ng]) for ng in c1 if ng in c2)
        total = sum(c1.values()) + sum(c2.values())
        return 2 * matched / total if total > 0 else 0.0

    def config(self) -> dict:
        return {"name": self.name, "n": self.n}


class PhoneticMetric(SyntacticSimilarityMetric):
    """Phonetic similarity via Soundex/Metaphone/NYSIIS word encoding.

    Encodes each word as a phonetic code, then computes Jaccard similarity on
    the encoded sets. Useful for archaic or variant spellings that sound alike:
    "baptize"/"baptise", "saith"/"said", "labour"/"labor".

    Requires jellyfish: pip install crossref[phonetic]
    """
    name = "phonetic"

    def __init__(self, algorithm: str = "soundex"):
        """
        Args:
            algorithm: One of 'soundex', 'metaphone', or 'nysiis'.
                'soundex'   — classic 4-char code; good for names
                'metaphone' — more aggressive reduction; better for common words
                'nysiis'    — New York State Intelligence System encoding
        """
        self.algorithm = algorithm

    def score(self, text1: str, text2: str) -> float:
        s1, s2 = self._encode_set(text1), self._encode_set(text2)
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        sets1 = [self._encode_set(t) for t in texts1]
        sets2 = [self._encode_set(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, s1 in enumerate(sets1):
            for j, s2 in enumerate(sets2):
                union = len(s1 | s2)
                scores[i, j] = len(s1 & s2) / union if union > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        sets = [self._encode_set(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                union = len(sets[i] | sets[j])
                scores[i, j] = scores[j, i] = len(sets[i] & sets[j]) / union if union > 0 else 1.0
        return scores

    def _encode_set(self, text: str) -> set:
        import jellyfish
        enc = getattr(jellyfish, self.algorithm)
        return {enc(w) for w in text.lower().split() if w.isalpha()}

    def config(self) -> dict:
        return {"name": self.name, "algorithm": self.algorithm}


class FuzzyNGramMetric(SyntacticSimilarityMetric):
    """Fuzzy string matching via rapidfuzz token_set_ratio.

    Computes Levenshtein similarity on the sorted token intersection plus
    remainders, making it invariant to word order and robust to partial
    matches — good for detecting quotations where words are reordered,
    added, or removed.

    Uses rapidfuzz's optimized C++ implementation for fast bulk scoring.

    Requires rapidfuzz: pip install crossref[fuzzy]
    """
    name = "fuzzy_ngram"

    def score(self, text1: str, text2: str) -> float:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(text1, text2) / 100.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        from rapidfuzz.process import cdist
        from rapidfuzz.fuzz import token_set_ratio
        return cdist(texts1, texts2, scorer=token_set_ratio) / 100.0

    def config(self) -> dict:
        return {"name": self.name}
