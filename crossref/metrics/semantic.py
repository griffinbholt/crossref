import logging

import numpy as np

from sentence_transformers import SentenceTransformer, SimilarityFunction, CrossEncoder

from .base import SimilarityMetric

logger = logging.getLogger(__name__)


class SemanticSimilarityMetric(SimilarityMetric):
    pass


class SentenceTransformerMetric(SemanticSimilarityMetric):
    """
    https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

    Best model: all-mpnet-base-v2
    Fastest model w/ close performance: all-MiniLM-L6-v2
    """
    name = "sentencetransformer"

    def __init__(self, modelname: str, similarity_fn: SimilarityFunction = SimilarityFunction.COSINE):
        super().__init__()
        self.modelname = modelname
        self.similarity_fn = similarity_fn
        logger.info("Loading SentenceTransformer model: %s", modelname)
        self.model = SentenceTransformer(modelname, similarity_fn_name=similarity_fn)
        logger.debug("Model loaded: %s", modelname)

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        logger.debug("Encoding %d + %d passages", len(texts1), len(texts2))
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        return self.model.similarity(embeddings1, embeddings2).detach().cpu().numpy()

    def score_self(self, texts: list[str]) -> np.ndarray:
        logger.debug("Encoding %d passages (self-comparison)", len(texts))
        embeddings = self.model.encode(texts)
        return self.model.similarity(embeddings, embeddings).detach().cpu().numpy()

    def config(self) -> dict:
        return {
            "name": self.name,
            "modelname": self.modelname,
            "similarity_fn": self.similarity_fn.value,
        }


class Doc2VecMetric(SemanticSimilarityMetric):
    """Doc2Vec document similarity using Gensim.

    https://radimrehurek.com/gensim/models/doc2vec.html
    https://spotintelligence.com/2023/09/06/doc2vec/

    Note: score_all() is the preferred entry point. It trains on the full
    corpus first, producing higher-quality embeddings. score() trains a fresh
    model on just the two texts, which is weaker but works for one-off use.
    """
    name = "doc2vec"

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 40,
        dm: int = 1,
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        tagged = [
            TaggedDocument(words=tokens1, tags=[0]),
            TaggedDocument(words=tokens2, tags=[1]),
        ]
        model = Doc2Vec(
            tagged,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            dm=self.dm,
        )
        return float(model.similarity_unseen_docs(tokens1, tokens2))

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        from gensim.models.doc2vec import TaggedDocument

        all_texts = texts1 + texts2
        tagged = [TaggedDocument(words=self._tokenize(t), tags=[i]) for i, t in enumerate(all_texts)]
        model = self._train(tagged)
        vecs1 = np.array([model.infer_vector(self._tokenize(t)) for t in texts1])
        vecs2 = np.array([model.infer_vector(self._tokenize(t)) for t in texts2])
        return self._cosine_matrix(vecs1, vecs2)

    def score_self(self, texts: list[str]) -> np.ndarray:
        from gensim.models.doc2vec import TaggedDocument

        tagged = [TaggedDocument(words=self._tokenize(t), tags=[i]) for i, t in enumerate(texts)]
        model = self._train(tagged)
        vecs = np.array([model.infer_vector(self._tokenize(t)) for t in texts])
        return self._cosine_matrix(vecs, vecs)

    def _train(self, tagged):
        logger.info("Doc2VecMetric: training on %d documents (%d epochs)", len(tagged), self.epochs)
        from gensim.models.doc2vec import Doc2Vec
        return Doc2Vec(
            tagged,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            dm=self.dm,
        )

    def _cosine_matrix(self, vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        norms1 = np.where((n := np.linalg.norm(vecs1, axis=1, keepdims=True)) == 0, 1.0, n)
        norms2 = np.where((n := np.linalg.norm(vecs2, axis=1, keepdims=True)) == 0, 1.0, n)
        return (vecs1 / norms1) @ (vecs2 / norms2).T

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def config(self) -> dict:
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "dm": self.dm,
            "workers": self.workers,
        }


class AveragedWordVecMetric(SemanticSimilarityMetric):
    """Semantic similarity via averaged pre-trained word vectors.

    Each passage is encoded as the mean of its word vectors from a pre-trained
    embedding model (Word2Vec, GloVe, FastText via gensim.downloader), then
    compared by cosine similarity.

    Unlike Doc2VecMetric, this requires no corpus training — just vector lookup
    and averaging. The pre-trained embeddings carry much richer general-language
    semantic knowledge (trained on billions of words), at the cost of potentially
    poor coverage for domain-specific vocabulary: archaic or proper-noun words
    like "Nephi", "saith", or "verily" may be out-of-vocabulary and are silently
    skipped. For scripture, Doc2VecMetric often has better coverage; this metric
    typically has better general semantic quality for common vocabulary.

    Recommended models (fast to slow, small to large):
        glove-wiki-gigaword-100     100d,  400k vocab, ~128 MB
        glove-wiki-gigaword-300     300d,  400k vocab, ~376 MB
        word2vec-google-news-300    300d, 3M vocab,    ~1.6 GB
        fasttext-wiki-news-subwords-300  300d, handles OOV via subwords

    For scripture use, fasttext is recommended: its subword decomposition
    approximates vectors for OOV words using character n-grams, so "Nephite"
    would borrow from "Neph..." subwords rather than being skipped entirely.

    Requires gensim: pip install crossref[doc2vec]
    """
    name = "avgwordvec"

    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        """
        Args:
            model_name: gensim.downloader model name. Downloaded automatically
                        on first use and cached locally.
        """
        self.model_name = model_name
        self._model = None  # lazy-loaded; excluded from pickle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_model'] = None  # don't pickle the loaded model; reload on demand
        return state

    def score(self, text1: str, text2: str) -> float:
        v1 = self._encode(text1)
        v2 = self._encode(text2)
        if v1 is None or v2 is None:
            return 0.0
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return float(np.dot(v1, v2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        vecs1 = self._encode_all(texts1)
        vecs2 = self._encode_all(texts2)
        return self._cosine_matrix(vecs1, vecs2)

    def score_self(self, texts: list[str]) -> np.ndarray:
        vecs = self._encode_all(texts)
        return self._cosine_matrix(vecs, vecs)

    def _cosine_matrix(self, vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        norms1 = np.where((n := np.linalg.norm(vecs1, axis=1, keepdims=True)) == 0, 1.0, n)
        norms2 = np.where((n := np.linalg.norm(vecs2, axis=1, keepdims=True)) == 0, 1.0, n)
        return (vecs1 / norms1) @ (vecs2 / norms2).T

    def _load_model(self):
        if self._model is None:
            logger.info("AveragedWordVecMetric: loading model %r", self.model_name)
            import gensim.downloader
            self._model = gensim.downloader.load(self.model_name)

    def _encode(self, text: str) -> np.ndarray | None:
        self._load_model()
        tokens = text.lower().split()
        vecs = [self._model[t] for t in tokens if t in self._model]
        return np.mean(vecs, axis=0) if vecs else None

    def _encode_all(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        dim = self._model.vector_size
        result = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            v = self._encode(text)
            if v is not None:
                result[i] = v
        return result

    def config(self) -> dict:
        return {"name": self.name, "model_name": self.model_name}


class ColBERTMetric(SemanticSimilarityMetric):
    """Late-interaction similarity using ColBERT.

    Unlike bi-encoders (one vector per passage), ColBERT produces one embedding
    per token. Similarity is computed via MaxSim: for each query token, find the
    most similar document token and sum those maxima across all query tokens.
    This captures fine-grained lexical and semantic alignment that a single
    pooled vector misses, at a cost between bi-encoder and cross-encoder.

    ColBERT is asymmetric (query tokens get a [Q] marker, document tokens [D]).
    This class symmetrizes by averaging both directions: (MaxSim(a→b) + MaxSim(b→a)) / 2.

    Scores are raw ColBERT MaxSim values (not bounded to [0, 1]).

    Recommended model: colbert-ir/colbertv2.0
    Requires pylate: pip install crossref[colbert]
    """
    name = "colbert"

    def __init__(self, modelname: str = "colbert-ir/colbertv2.0"):
        self.modelname = modelname
        self._model = None  # lazy-loaded; excluded from pickle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        from pylate import scores as pylate_scores
        self._load_model()
        emb1_q = self._model.encode(texts1, is_query=True)
        emb2_d = self._model.encode(texts2, is_query=False)
        emb2_q = self._model.encode(texts2, is_query=True)
        emb1_d = self._model.encode(texts1, is_query=False)
        fwd = np.array(pylate_scores.colbert_scores(emb1_q, emb2_d))  # (n1, n2)
        bwd = np.array(pylate_scores.colbert_scores(emb2_q, emb1_d))  # (n2, n1)
        return (fwd + bwd.T) / 2

    def score_self(self, texts: list[str]) -> np.ndarray:
        from pylate import scores as pylate_scores
        self._load_model()
        embs_q = self._model.encode(texts, is_query=True)
        embs_d = self._model.encode(texts, is_query=False)
        fwd = np.array(pylate_scores.colbert_scores(embs_q, embs_d))  # (n, n)
        return (fwd + fwd.T) / 2

    def _load_model(self):
        if self._model is None:
            logger.info("ColBERTMetric: loading model %r", self.modelname)
            from pylate import models
            self._model = models.ColBERT(model_name_or_path=self.modelname)

    def config(self) -> dict:
        return {"name": self.name, "modelname": self.modelname}


class LLMJudgeMetric(SemanticSimilarityMetric):
    """LLM-as-judge semantic similarity.

    Prompts a large language model to rate the semantic similarity of two passages
    on a scale from 0.0 to 1.0. Captures reasoning-level equivalence — paraphrase,
    logical entailment, thematic overlap — that embedding-based metrics miss.

    Significantly slower and more expensive than other metrics (one API call per
    unique pair). Best used as a reference score or ensemble component, not as
    the primary metric for large corpora. score_self() skips the diagonal
    (self-similarity = 1.0) and queries only the upper triangle.

    Supported providers:
        "anthropic"  — Claude models (e.g. "claude-haiku-4-5-20251001")
        "openai"     — OpenAI models (e.g. "gpt-4o-mini")

    API key is read from ANTHROPIC_API_KEY / OPENAI_API_KEY environment variables,
    or passed explicitly as api_key.

    Requires: pip install crossref[llm]
    """
    name = "llm_judge"

    _DEFAULT_PROMPT = (
        "Rate the semantic similarity of the following two passages on a scale "
        "from 0.0 to 1.0, where 0.0 means completely unrelated and 1.0 means "
        "identical in meaning. Consider conceptual overlap, paraphrase, and "
        "shared themes. Respond with only a single decimal number and nothing else.\n\n"
        "Passage 1: {text1}\n\n"
        "Passage 2: {text2}\n\n"
        "Similarity score:"
    )

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        prompt_template: str | None = None,
        workers: int = 1,
    ):
        """
        Args:
            provider:        "anthropic" or "openai"
            model:           Model name, e.g. "claude-haiku-4-5-20251001" or "gpt-4o-mini"
            api_key:         API key. If None, reads from the provider's standard env var.
            prompt_template: Optional custom prompt. Must contain {text1} and {text2}.
            workers:         Number of concurrent threads for parallel API calls.
                             Each pair requires one API call; workers > 1 issues them
                             concurrently via ThreadPoolExecutor. Check your API rate
                             limits before setting this high.
        """
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider must be 'anthropic' or 'openai', got {provider!r}")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT
        self.workers = workers

    def score(self, text1: str, text2: str) -> float:
        prompt = self.prompt_template.format(text1=text1, text2=text2)
        return self._parse_score(self._call_llm(prompt))

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        n_pairs = len(texts1) * len(texts2)
        logger.info(
            "LLMJudgeMetric.score_all: %d pairs (%dx%d) via %s/%s",
            n_pairs, len(texts1), len(texts2), self.provider, self.model,
        )
        pairs = [(t1, t2) for t1 in texts1 for t2 in texts2]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                flat = list(ex.map(lambda p: self.score(*p), pairs))
        else:
            flat = [self.score(*p) for p in pairs]
        return np.array(flat).reshape(len(texts1), len(texts2))

    def score_self(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        n_pairs = n * (n - 1) // 2
        logger.info(
            "LLMJudgeMetric.score_self: %d pairs (%d passages) via %s/%s",
            n_pairs, n, self.provider, self.model,
        )
        scores = np.ones((n, n))
        pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(lambda p: self.score(texts[p[0]], texts[p[1]]), pairs_ij))
        else:
            results = [self.score(texts[i], texts[j]) for i, j in pairs_ij]
        for (i, j), s in zip(pairs_ij, results):
            scores[i, j] = scores[j, i] = s
        return scores

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        else:  # openai
            import openai
            client = openai.OpenAI(api_key=self.api_key) if self.api_key else openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

    def _parse_score(self, response: str) -> float:
        import re
        match = re.search(r'\b(1(\.0*)?|0(\.\d+)?)\b', response.strip())
        return float(match.group()) if match else 0.0

    def config(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            # api_key intentionally excluded for security
            "prompt_template": self.prompt_template,
            "workers": self.workers,
        }


class APIEmbeddingMetric(SemanticSimilarityMetric):
    """Semantic similarity via cloud embedding APIs.

    Encodes passages using a cloud embedding API, then computes cosine similarity.
    Functionally equivalent to SentenceTransformerMetric but uses a remote API
    instead of a local model — no GPU required, but requires network access and
    an API key. Useful for accessing the largest proprietary embedding models.

    Supported providers and example models:
        "openai"  — "text-embedding-3-large", "text-embedding-3-small"
        "cohere"  — "embed-v4.0", "embed-english-v3.0"
        "voyage"  — "voyage-3", "voyage-3-lite"

    API key is read from the provider's standard environment variable
    (OPENAI_API_KEY, CO_API_KEY, VOYAGE_API_KEY), or passed as api_key.

    Requires: pip install crossref[llm]
    """
    name = "api_embedding"

    def __init__(self, provider: str, model: str, api_key: str | None = None):
        """
        Args:
            provider: "openai", "cohere", or "voyage"
            model:    Embedding model name
            api_key:  API key. If None, reads from the provider's standard env var.
        """
        if provider not in ("openai", "cohere", "voyage"):
            raise ValueError(f"provider must be 'openai', 'cohere', or 'voyage', got {provider!r}")
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        vecs1 = self._embed(texts1)
        vecs2 = self._embed(texts2)
        return self._cosine_matrix(vecs1, vecs2)

    def score_self(self, texts: list[str]) -> np.ndarray:
        vecs = self._embed(texts)
        return self._cosine_matrix(vecs, vecs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        logger.debug("APIEmbeddingMetric: embedding %d texts via %s/%s", len(texts), self.provider, self.model)
        if self.provider == "openai":
            import openai
            client = openai.OpenAI(api_key=self.api_key) if self.api_key else openai.OpenAI()
            resp = client.embeddings.create(input=texts, model=self.model)
            return np.array([d.embedding for d in resp.data])
        elif self.provider == "cohere":
            import cohere
            client = cohere.ClientV2(api_key=self.api_key) if self.api_key else cohere.ClientV2()
            resp = client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
            )
            return np.array(resp.embeddings.float_)
        else:  # voyage
            import voyageai
            client = voyageai.Client(api_key=self.api_key) if self.api_key else voyageai.Client()
            resp = client.embed(texts, model=self.model)
            return np.array(resp.embeddings)

    def _cosine_matrix(self, vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        norms1 = np.where((n := np.linalg.norm(vecs1, axis=1, keepdims=True)) == 0, 1.0, n)
        norms2 = np.where((n := np.linalg.norm(vecs2, axis=1, keepdims=True)) == 0, 1.0, n)
        return (vecs1 / norms1) @ (vecs2 / norms2).T

    def config(self) -> dict:
        return {"name": self.name, "provider": self.provider, "model": self.model}


class CrossEncoderMetric(SemanticSimilarityMetric):
    """Cross-encoder similarity using sentence-transformers CrossEncoder.

    Unlike bi-encoders (SentenceTransformerMetric), cross-encoders take both
    texts as a single joint input, allowing attention to flow between them.
    This produces higher-quality scores at the cost of speed: there are no
    reusable per-text embeddings, so every pair requires a full forward pass.

    Raw cross-encoder scores are asymmetric (order of texts matters). This
    class symmetrizes by averaging both directions: (score(a,b) + score(b,a)) / 2.

    Recommended STS models:
        cross-encoder/stsb-roberta-large   (highest quality)
        cross-encoder/stsb-distilroberta-base  (faster)

    No additional dependencies — uses the same sentence_transformers package
    as SentenceTransformerMetric.
    """
    name = "crossencoder"

    def __init__(self, modelname: str):
        self.modelname = modelname
        logger.info("Loading CrossEncoder model: %s", modelname)
        self.model = CrossEncoder(modelname)

    def score(self, text1: str, text2: str) -> float:
        fwd = float(self.model.predict([(text1, text2)])[0])
        bwd = float(self.model.predict([(text2, text1)])[0])
        return (fwd + bwd) / 2

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        pairs_fwd = [(t1, t2) for t1 in texts1 for t2 in texts2]
        pairs_bwd = [(t2, t1) for t1 in texts1 for t2 in texts2]
        fwd = self.model.predict(pairs_fwd).reshape(len(texts1), len(texts2))
        bwd = self.model.predict(pairs_bwd).reshape(len(texts1), len(texts2))
        return (fwd + bwd) / 2

    def score_self(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        matrix = np.zeros((n, n))
        # Diagonal: score(t, t) is the same in both directions
        if n > 0:
            diag = self.model.predict([(t, t) for t in texts])
            np.fill_diagonal(matrix, diag)
        # Upper triangle: average both directions per pair
        pairs_fwd = [(texts[i], texts[j]) for i in range(n) for j in range(i + 1, n)]
        if pairs_fwd:
            pairs_bwd = [(texts[j], texts[i]) for i in range(n) for j in range(i + 1, n)]
            scores = (self.model.predict(pairs_fwd) + self.model.predict(pairs_bwd)) / 2
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i, j] = matrix[j, i] = scores[k]
                    k += 1
        return matrix

    def config(self) -> dict:
        return {"name": self.name, "modelname": self.modelname}


class WMDMetric(SemanticSimilarityMetric):
    """Word Mover's Distance semantic similarity.

    Computes the minimum "transport cost" to move all word mass from one
    passage to the other, using word embedding distances as transport costs.
    Captures semantic similarity between passages using different but
    synonymous vocabulary.

    Similarity = 1 / (1 + WMD), where WMD is gensim's normalized distance.

    Note: WMD is O(n³) per pair via linear programming — much slower than
    other metrics. Not recommended for corpora with many long passages.

    Requires gensim: pip install crossref[doc2vec]
    """
    name = "wmd"

    def __init__(self, model_name: str = "glove-wiki-gigaword-100", workers: int = 1):
        """
        Args:
            model_name: gensim.downloader model name.
            workers:    Number of concurrent threads. WMD is O(n³) per pair via LP,
                        so parallelism provides near-linear speedup. Gensim's
                        KeyedVectors.wmdistance() releases the GIL, making
                        ThreadPoolExecutor effective.
        """
        self.model_name = model_name
        self.workers = workers
        self._model = None  # lazy-loaded; excluded from pickle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def score(self, text1: str, text2: str) -> float:
        self._load_model()
        t1, t2 = text1.lower().split(), text2.lower().split()
        if not t1 or not t2:
            return 0.0
        dist = self._model.wmdistance(t1, t2)
        return float(1.0 / (1.0 + dist)) if dist != float('inf') else 0.0

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        logger.debug("WMDMetric.score_all: %dx%d (O(n³) per pair)", len(texts1), len(texts2))
        self._load_model()  # pre-load before threads to avoid race on first access
        pairs = [(t1, t2) for t1 in texts1 for t2 in texts2]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                flat = list(ex.map(lambda p: self.score(*p), pairs))
        else:
            flat = [self.score(*p) for p in pairs]
        return np.array(flat).reshape(len(texts1), len(texts2))

    def score_self(self, texts: list[str]) -> np.ndarray:
        self._load_model()  # pre-load before threads to avoid race on first access
        n = len(texts)
        scores = np.ones((n, n))
        pairs_ij = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(lambda p: self.score(texts[p[0]], texts[p[1]]), pairs_ij))
        else:
            results = [self.score(texts[i], texts[j]) for i, j in pairs_ij]
        for (i, j), s in zip(pairs_ij, results):
            scores[i, j] = scores[j, i] = s
        return scores

    def _load_model(self):
        if self._model is None:
            logger.info("WMDMetric: loading model %r", self.model_name)
            import gensim.downloader
            self._model = gensim.downloader.load(self.model_name)

    def config(self) -> dict:
        return {"name": self.name, "model_name": self.model_name, "workers": self.workers}


class SoftCosineMetric(SemanticSimilarityMetric):
    """Soft Cosine Similarity using pre-trained word embeddings.

    Like TF-IDF cosine but gives partial credit for semantically similar words:
    the kernel matrix S[i,j] = cosine_sim(embed(w_i), embed(w_j)) propagates
    similarity across vocabulary entries, so "baptism" partially matches
    "immersion" without explicit synonym lists.

    Bridges syntactic and semantic similarity: interpretable word-level term
    frequency counts with pre-trained embedding knowledge.

    Requires gensim: pip install crossref[doc2vec]
    """
    name = "soft_cosine"

    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        self.model_name = model_name
        self._model = None  # lazy-loaded; excluded from pickle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def score(self, text1: str, text2: str) -> float:
        return float(self.score_all([text1], [text2])[0, 0])

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        self._load_model()
        vecs1, S = self._vectorize(texts1, texts1 + texts2)
        vecs2, _ = self._vectorize(texts2, texts1 + texts2)
        return self._soft_cosine_matrix(vecs1, vecs2, S)

    def score_self(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        vecs, S = self._vectorize(texts, texts)
        return self._soft_cosine_matrix(vecs, vecs, S)

    def _load_model(self):
        if self._model is None:
            logger.info("SoftCosineMetric: loading model %r", self.model_name)
            import gensim.downloader
            self._model = gensim.downloader.load(self.model_name)

    def _vectorize(self, texts: list[str], corpus: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Build TF vectors and word similarity matrix from the corpus vocabulary."""
        all_tokens = [t.lower().split() for t in corpus]
        vocab = sorted({w for tokens in all_tokens for w in tokens if w in self._model})
        if not vocab:
            return np.zeros((len(texts), 0)), np.zeros((0, 0))
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        E = np.array([self._model[w] for w in vocab])
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E_norm = E / np.where(norms == 0, 1.0, norms)
        S = E_norm @ E_norm.T  # (V, V) pairwise cosine similarities
        vecs = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            for w in text.lower().split():
                if w in vocab_idx:
                    vecs[i, vocab_idx[w]] += 1.0
        return vecs, S

    def _soft_cosine_matrix(self, vecs1: np.ndarray, vecs2: np.ndarray, S: np.ndarray) -> np.ndarray:
        if S.size == 0:
            return np.zeros((len(vecs1), len(vecs2)))
        numerator = (vecs1 @ S) @ vecs2.T
        sv1 = np.einsum('ij,jk,ik->i', vecs1, S, vecs1)
        sv2 = np.einsum('ij,jk,ik->i', vecs2, S, vecs2)
        denom = np.sqrt(np.maximum(sv1[:, np.newaxis] * sv2[np.newaxis, :], 0.0))
        return numerator / np.where(denom == 0, 1.0, denom)

    def config(self) -> dict:
        return {"name": self.name, "model_name": self.model_name}
