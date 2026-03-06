from .base import SimilarityMetric
from .syntactic import (
    SyntacticSimilarityMetric,
    NGramMetric, JaccardMetric, CharNGramMetric,
    LCSMetric, LevenshteinMetric, FuzzyNGramMetric,
    LCSubstringMetric, DiceMetric, ROUGELMetric, ROUGENMetric, PhoneticMetric,
)
from .semantic import (
    SemanticSimilarityMetric,
    SentenceTransformerMetric, CrossEncoderMetric,
    ColBERTMetric, LLMJudgeMetric, APIEmbeddingMetric,
    Doc2VecMetric, AveragedWordVecMetric,
    WMDMetric, SoftCosineMetric,
)
from .statistical import (
    StatisticalSimilarityMetric,
    NCDMetric, TFIDFMetric, BM25Metric,
    LSAMetric, JSMetric, LZJDMetric,
)


def _build_sentencetransformer(c: dict) -> SentenceTransformerMetric:
    from sentence_transformers import SimilarityFunction
    return SentenceTransformerMetric(
        modelname=c['modelname'],
        similarity_fn=SimilarityFunction(c.get('similarity_fn', 'cosine')),
    )


_BUILDERS = {
    # Syntactic
    NGramMetric.name: lambda c: NGramMetric(
        n_min=c.get('n_min', 2),
        n_max=c.get('n_max'),
        custom_stopphrases=set(c.get('custom_stopphrases', [])),
        workers=c.get('workers', 1),
    ),
    JaccardMetric.name: lambda c: JaccardMetric(n=c.get('n', 1)),
    CharNGramMetric.name: lambda c: CharNGramMetric(n=c.get('n', 4)),
    LCSMetric.name: lambda c: LCSMetric(workers=c.get('workers', 1)),
    LevenshteinMetric.name: lambda c: LevenshteinMetric(workers=c.get('workers', 1)),
    FuzzyNGramMetric.name: lambda c: FuzzyNGramMetric(),
    LCSubstringMetric.name: lambda c: LCSubstringMetric(workers=c.get('workers', 1)),
    DiceMetric.name: lambda c: DiceMetric(n=c.get('n', 1)),
    ROUGELMetric.name: lambda c: ROUGELMetric(workers=c.get('workers', 1)),
    ROUGENMetric.name: lambda c: ROUGENMetric(n=c.get('n', 1)),
    PhoneticMetric.name: lambda c: PhoneticMetric(algorithm=c.get('algorithm', 'soundex')),
    # Semantic
    SentenceTransformerMetric.name: _build_sentencetransformer,
    CrossEncoderMetric.name: lambda c: CrossEncoderMetric(modelname=c['modelname']),
    ColBERTMetric.name: lambda c: ColBERTMetric(modelname=c.get('modelname', 'colbert-ir/colbertv2.0')),
    LLMJudgeMetric.name: lambda c: LLMJudgeMetric(
        provider=c['provider'],
        model=c['model'],
        api_key=c.get('api_key'),
        prompt_template=c.get('prompt_template'),
        workers=c.get('workers', 1),
    ),
    APIEmbeddingMetric.name: lambda c: APIEmbeddingMetric(
        provider=c['provider'],
        model=c['model'],
        api_key=c.get('api_key'),
    ),
    Doc2VecMetric.name: lambda c: Doc2VecMetric(
        vector_size=c.get('vector_size', 100),
        window=c.get('window', 5),
        min_count=c.get('min_count', 1),
        epochs=c.get('epochs', 40),
        dm=c.get('dm', 1),
        workers=c.get('workers', 4),
    ),
    AveragedWordVecMetric.name: lambda c: AveragedWordVecMetric(
        model_name=c.get('model_name', 'glove-wiki-gigaword-100'),
    ),
    WMDMetric.name: lambda c: WMDMetric(
        model_name=c.get('model_name', 'glove-wiki-gigaword-100'),
        workers=c.get('workers', 1),
    ),
    SoftCosineMetric.name: lambda c: SoftCosineMetric(model_name=c.get('model_name', 'glove-wiki-gigaword-100')),
    # Statistical
    NCDMetric.name: lambda c: NCDMetric(level=c.get('level', 9), workers=c.get('workers', 1)),
    TFIDFMetric.name: lambda c: TFIDFMetric(
        analyzer=c.get('analyzer', 'word'),
        ngram_range=tuple(c.get('ngram_range', [1, 1])),
        max_features=c.get('max_features'),
        sublinear_tf=c.get('sublinear_tf', True),
    ),
    BM25Metric.name: lambda c: BM25Metric(k1=c.get('k1', 1.5), b=c.get('b', 0.75)),
    LSAMetric.name: lambda c: LSAMetric(
        n_components=c.get('n_components', 100),
        ngram_range=tuple(c.get('ngram_range', [1, 1])),
    ),
    JSMetric.name: lambda c: JSMetric(),
    LZJDMetric.name: lambda c: LZJDMetric(),
}


def metric_from_config(config: dict) -> SimilarityMetric:
    """Reconstruct a SimilarityMetric from a config dict produced by metric.config().

    Used internally to restore metrics after loading saved Results, and by the
    CLI to instantiate metrics from a YAML config file.
    """
    name = config.get('name')
    if not name:
        raise ValueError("Metric config must include a 'name' field.")
    if name not in _BUILDERS:
        raise NotImplementedError(
            f"Unknown metric: {name!r}. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](config)
