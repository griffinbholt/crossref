import logging
import zlib

import numpy as np

from .base import SimilarityMetric

logger = logging.getLogger(__name__)


class StatisticalSimilarityMetric(SimilarityMetric):
    pass


class NCDMetric(StatisticalSimilarityMetric):
    """Normalized Compression Distance similarity.

    Uses zlib compression to measure shared information between two texts.
    If two passages share structure or content, compressing them together
    is cheaper than compressing them separately — NCD captures that redundancy.

    NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    where C(x) = compressed byte length of x. NCD is in [0, 1] where 0 means
    identical and 1 means nothing in common. This class returns 1 - NCD so
    that higher scores mean more similar, consistent with all other metrics.

    Language-agnostic and requires no training or external dependencies.
    """
    name = "ncd"

    def __init__(self, level: int = 9, workers: int = 1):
        """
        Args:
            level:   zlib compression level (1–9). Higher = smaller output,
                     more accurate NCD, but slower. Default 9 (maximum).
            workers: Number of concurrent threads. zlib.compress() releases
                     the GIL, so ThreadPoolExecutor provides real speedup.
        """
        self.level = level
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        b1 = text1.encode()
        b2 = text2.encode()
        c1 = len(zlib.compress(b1, self.level))
        c2 = len(zlib.compress(b2, self.level))
        c12 = len(zlib.compress(b1 + b2, self.level))
        denom = max(c1, c2)
        if denom == 0:
            return 0.0
        ncd = (c12 - min(c1, c2)) / denom
        return float(1.0 - ncd)

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        logger.debug("NCDMetric.score_all: %dx%d (level=%d)", len(texts1), len(texts2), self.level)
        encoded1 = [t.encode() for t in texts1]
        encoded2 = [t.encode() for t in texts2]
        c1 = [len(zlib.compress(b, self.level)) for b in encoded1]
        c2 = [len(zlib.compress(b, self.level)) for b in encoded2]
        pairs = [(b1, ci, b2, cj) for b1, ci in zip(encoded1, c1) for b2, cj in zip(encoded2, c2)]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                flat = list(ex.map(lambda p: self._ncd(*p), pairs))
        else:
            flat = [self._ncd(*p) for p in pairs]
        return np.array(flat).reshape(len(texts1), len(texts2))

    def score_self(self, texts: list[str]) -> np.ndarray:
        encoded = [t.encode() for t in texts]
        c = [len(zlib.compress(b, self.level)) for b in encoded]
        n = len(texts)
        scores = np.ones((n, n))
        pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(
                    lambda p: self._ncd(encoded[p[0]], c[p[0]], encoded[p[1]], c[p[1]]),
                    pairs_ij,
                ))
        else:
            results = [self._ncd(encoded[i], c[i], encoded[j], c[j]) for i, j in pairs_ij]
        for (i, j), s in zip(pairs_ij, results):
            scores[i, j] = scores[j, i] = s
        return scores

    def _ncd(self, b1: bytes, c1: int, b2: bytes, c2: int) -> float:
        c12 = len(zlib.compress(b1 + b2, self.level))
        denom = max(c1, c2)
        return 1.0 - (c12 - min(c1, c2)) / denom if denom > 0 else 0.0

    def config(self) -> dict:
        return {"name": self.name, "level": self.level, "workers": self.workers}


class TFIDFMetric(StatisticalSimilarityMetric):
    """TF-IDF cosine similarity.

    Fits a TF-IDF vectorizer on the full corpus, then computes cosine similarity
    between passage vectors. IDF weights down common words across all passages
    and up words that are distinctive to a passage — complementary to NGramMetric,
    which treats all word matches equally.

    score_all() is the preferred entry point: IDF weights are only meaningful
    relative to the full corpus. score() fits on just two texts, so IDF is
    degenerate (all words appear in both → IDF ≈ 0), making it nearly useless
    for single-pair queries.

    Requires scikit-learn: pip install crossref[tfidf]
    """
    name = "tfidf"

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        sublinear_tf: bool = True,
    ):
        """
        Args:
            analyzer:     'word' or 'char'. 'word' is standard; 'char' can help
                          with spelling variants across translations.
            ngram_range:  Range of n-gram sizes for the vocabulary, e.g. (1, 2)
                          includes unigrams and bigrams.
            max_features: Cap vocabulary size. None = keep all terms.
            sublinear_tf: Apply log(1 + tf) scaling. Reduces the dominance of
                          very frequent terms. Recommended for short passages.
        """
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        vectorizer = self._fit_vectorizer(texts1 + texts2)
        vecs1 = vectorizer.transform(texts1)
        vecs2 = vectorizer.transform(texts2)
        return (vecs1 @ vecs2.T).toarray()

    def score_self(self, texts: list[str]) -> np.ndarray:
        vecs = self._fit_vectorizer(texts).transform(texts)
        return (vecs @ vecs.T).toarray()

    def _fit_vectorizer(self, texts: list[str]):
        logger.debug("TFIDFMetric: fitting vectorizer on %d texts", len(texts))
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            norm="l2",  # unit vectors → dot product = cosine similarity
        )
        vectorizer.fit(texts)
        return vectorizer

    def config(self) -> dict:
        return {
            "name": self.name,
            "analyzer": self.analyzer,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
        }


class BM25Metric(StatisticalSimilarityMetric):
    """BM25 (Okapi BM25) similarity.

    An improvement over TF-IDF with term frequency saturation and document
    length normalization — the standard ranking function behind most search
    engines.

    Raw BM25 is asymmetric (query → ranked documents). Since cross-referencing
    implies symmetric similarity, score_all() symmetrizes by averaging both
    directions: (BM25(i→j) + BM25(j→i)) / 2, then normalizes by the global
    maximum so that 1.0 = strongest match in the entire comparison.

    Requires rank-bm25: pip install crossref[bm25]
    """
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation. Controls how quickly additional
                occurrences of a term stop adding score. Typical range 1.2–2.0.
            b:  Document length normalization. 1.0 = full normalization by
                length, 0.0 = no normalization. Default 0.75.
        """
        self.k1 = k1
        self.b = b

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        logger.debug("BM25Metric.score_all: %dx%d", len(texts1), len(texts2))
        from rank_bm25 import BM25Okapi
        tokenized1 = [self._tokenize(t) for t in texts1]
        tokenized2 = [self._tokenize(t) for t in texts2]
        fwd = np.array([BM25Okapi(tokenized2, k1=self.k1, b=self.b).get_scores(t) for t in tokenized1])
        bwd = np.array([BM25Okapi(tokenized1, k1=self.k1, b=self.b).get_scores(t) for t in tokenized2])
        scores = (fwd + bwd.T) / 2
        global_max = scores.max()
        return scores / global_max if global_max > 0 else scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        fwd = np.array([bm25.get_scores(t) for t in tokenized])
        scores = (fwd + fwd.T) / 2  # symmetrize with single index
        global_max = scores.max()
        return scores / global_max if global_max > 0 else scores

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def config(self) -> dict:
        return {"name": self.name, "k1": self.k1, "b": self.b}


class LSAMetric(StatisticalSimilarityMetric):
    """Latent Semantic Analysis (LSA) cosine similarity.

    Applies TruncatedSVD to the TF-IDF matrix, projecting passages into a
    low-dimensional latent topic space. Unlike TF-IDF cosine (exact term
    overlap), LSA discovers that passages using different words for the same
    topic — "baptism" vs "immersion" — are similar because those words
    co-occur with the same neighbors across the corpus.

    n_components controls the number of latent dimensions. Values in [50, 300]
    are typical; more dimensions capture finer distinctions but may overfit.

    Requires scikit-learn: pip install crossref[tfidf]
    """
    name = "lsa"

    def __init__(self, n_components: int = 100, ngram_range: tuple[int, int] = (1, 1)):
        self.n_components = n_components
        self.ngram_range = ngram_range

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        vecs = self._fit_transform(texts1 + texts2)
        return vecs[:len(texts1)] @ vecs[len(texts1):].T

    def score_self(self, texts: list[str]) -> np.ndarray:
        vecs = self._fit_transform(texts)
        return vecs @ vecs.T

    def _fit_transform(self, texts: list[str]) -> np.ndarray:
        logger.debug("LSAMetric: fitting on %d texts (n_components=%d)", len(texts), self.n_components)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        n_components = min(self.n_components, len(texts) - 1)
        if n_components < 1:
            return np.zeros((len(texts), 1))
        tfidf = TfidfVectorizer(ngram_range=self.ngram_range).fit_transform(texts)
        lsa = TruncatedSVD(n_components=n_components).fit_transform(tfidf)
        return normalize(lsa)  # L2 → dot product = cosine similarity

    def config(self) -> dict:
        return {"name": self.name, "n_components": self.n_components, "ngram_range": list(self.ngram_range)}


class JSMetric(StatisticalSimilarityMetric):
    """Jensen-Shannon divergence similarity on word frequency distributions.

    Treats each passage as a unigram probability distribution over words.
    JS divergence is the symmetric, smoothed version of KL divergence.

    Similarity = 1 - JS_distance(p, q), where JS distance = sqrt(JSD) ∈ [0,1].

    Captures distributional similarity: passages discussing the same topic
    in completely different words score high if their word distributions are
    similar. Works without any corpus or model — pure frequency statistics.
    """
    name = "js"

    def score(self, text1: str, text2: str) -> float:
        return 1.0 - self._js_distance(self._distribution(text1), self._distribution(text2))

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        dists1 = [self._distribution(t) for t in texts1]
        dists2 = [self._distribution(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, p in enumerate(dists1):
            for j, q in enumerate(dists2):
                scores[i, j] = 1.0 - self._js_distance(p, q)
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        dists = [self._distribution(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                scores[i, j] = scores[j, i] = 1.0 - self._js_distance(dists[i], dists[j])
        return scores

    def _distribution(self, text: str) -> dict[str, float]:
        from collections import Counter
        tokens = text.lower().split()
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {w: c / total for w, c in counts.items()}

    def _js_distance(self, p: dict, q: dict) -> float:
        """Jensen-Shannon distance = sqrt(JSD / ln2) ∈ [0, 1]."""
        vocab = set(p) | set(q)
        if not vocab:
            return 0.0
        jsd = 0.0
        for w in vocab:
            pw, qw = p.get(w, 0.0), q.get(w, 0.0)
            mw = (pw + qw) / 2
            if pw > 0:
                jsd += pw * np.log(pw / mw) / 2
            if qw > 0:
                jsd += qw * np.log(qw / mw) / 2
        return float(np.sqrt(max(0.0, jsd / np.log(2))))

    def config(self) -> dict:
        return {"name": self.name}


class LZJDMetric(StatisticalSimilarityMetric):
    """Lempel-Ziv Jaccard Distance similarity.

    Parses each text using LZ77-style incremental parsing into a set of
    unique substrings (phrases), then computes Jaccard similarity on those
    sets. More interpretable than NCD (it's a literal set operation) and
    empirically more accurate on short texts.

    No external dependencies — uses only stdlib.
    """
    name = "lzjd"

    def score(self, text1: str, text2: str) -> float:
        s1, s2 = self._lz_set(text1), self._lz_set(text2)
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 1.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        sets1 = [self._lz_set(t) for t in texts1]
        sets2 = [self._lz_set(t) for t in texts2]
        scores = np.zeros((len(texts1), len(texts2)))
        for i, s1 in enumerate(sets1):
            for j, s2 in enumerate(sets2):
                union = len(s1 | s2)
                scores[i, j] = len(s1 & s2) / union if union > 0 else 1.0
        return scores

    def score_self(self, texts: list[str]) -> np.ndarray:
        sets = [self._lz_set(t) for t in texts]
        n = len(texts)
        scores = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                union = len(sets[i] | sets[j])
                scores[i, j] = scores[j, i] = len(sets[i] & sets[j]) / union if union > 0 else 1.0
        return scores

    def _lz_set(self, text: str) -> frozenset:
        """LZ77-style incremental parsing into a frozenset of phrases."""
        phrases = set()
        seen = {''}
        current = ''
        for char in text:
            nxt = current + char
            if nxt in seen:
                current = nxt
            else:
                phrases.add(nxt)
                seen.add(nxt)
                current = ''
        if current:
            phrases.add(current)
        return frozenset(phrases)

    def config(self) -> dict:
        return {"name": self.name}
