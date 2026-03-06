# CrossRef

[![Tests](https://github.com/griffinbholt/crossref/actions/workflows/tests.yml/badge.svg)](https://github.com/griffinbholt/crossref/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CrossRef is a Python library and CLI for automated document cross-referencing. Given any two documents, it finds textually similar passage pairs using a suite of syntactic, semantic, and statistical similarity metrics.

**Primary use case:** comparing scriptural texts (Book of Mormon, KJV Bible, D&C, JST, etc.) — but it works on any plain text, Markdown, or structured document format.

---

## Installation

```bash
pip install crossref
```

Install optional dependency groups as needed:

```bash
pip install "crossref[hdf5]"      # HDF5 storage backend (h5py)
pip install "crossref[tfidf]"     # TFIDFMetric, LSAMetric (scikit-learn)
pip install "crossref[bm25]"      # BM25Metric (rank-bm25)
pip install "crossref[fuzzy]"     # FuzzyNGramMetric (rapidfuzz)
pip install "crossref[phonetic]"  # PhoneticMetric (jellyfish)
pip install "crossref[doc2vec]"   # Doc2VecMetric, WMDMetric, etc. (gensim)
pip install "crossref[colbert]"   # ColBERTMetric (pylate)
pip install "crossref[llm]"       # LLMJudgeMetric, APIEmbeddingMetric
pip install "crossref[pdf]"       # Document.from_file for .pdf (pypdf)
pip install "crossref[docx]"      # Document.from_file for .docx (python-docx)
pip install "crossref[epub]"      # Document.from_file for .epub (ebooklib)
```

---

## Quick Start

### Python API

```python
import crossref

# Load documents
bom = crossref.Document.from_file("book_of_mormon.md", splitter="line")
kjv = crossref.Document.from_file("kjv_new_testament.md", splitter="line")

# Choose metrics
metrics = [
    crossref.NGramMetric(n_min=2, n_max=10),
    crossref.SentenceTransformerMetric("all-MiniLM-L6-v2"),
]

# Compare
results = crossref.compare(bom, kjv, metrics=metrics)

# Explore results
results.top_k(0, metric="ngram", k=10)           # top 10 matches for passage 0
results.filter(metric="ngram", threshold=0.5)     # all pairs above 0.5
results.heatmap(metric="ngram", output="heatmap.png")
results.save("bom_kjv.pkl")

# Load later
results = crossref.load("bom_kjv.pkl")
```

### CLI

```bash
crossref config.yaml --output results.pkl
```

---

## Loading Documents

### From a file

```python
# Plain text — splits by line, paragraph, sentence, or section
doc = crossref.Document.from_file("text.txt", splitter="line")
doc = crossref.Document.from_file("text.txt", splitter="paragraph")
doc = crossref.Document.from_file("text.txt", splitter="sentence")
doc = crossref.Document.from_file("text.txt", splitter="section")

# Markdown — heading hierarchy becomes passage labels
doc = crossref.Document.from_file("book.md", splitter="line")

# Directory — walks recursively; subpath becomes label
doc = crossref.Document.from_file("my_book/", splitter="line")

# Other formats (require optional dependencies)
doc = crossref.Document.from_file("paper.pdf")
doc = crossref.Document.from_file("paper.docx")
doc = crossref.Document.from_file("book.epub")
```

### From structured data

```python
doc = crossref.Document.from_passages(
    ["In the beginning...", "And God said..."],
    labels=["Gen 1:1", "Gen 1:3"],
    title="Genesis",
)

doc = crossref.Document.from_csv("verses.csv", text_column="text", label_column="ref")
doc = crossref.Document.from_json("verses.json", text_key="text", label_key="ref")
doc = crossref.Document.from_jsonl("verses.jsonl", text_key="text", label_key="ref")
```

### Splitters

| Splitter | Splits on |
|---|---|
| `"line"` (default) | newlines |
| `"paragraph"` | blank lines |
| `"section"` | Markdown headings (`#`) |
| `"sentence"` | sentence boundaries (NLTK) |
| callable | custom function `str → list[str]` |

---

## Comparing Documents

```python
# Single pair → Results
results = crossref.compare(doc_a, doc_b, metrics=metrics)

# Self-comparison → Results  (uses score_self for efficiency)
results = crossref.compare(doc_a, metrics=metrics)

# All-vs-all → ResultsCollection
results = crossref.compare([doc_a, doc_b, doc_c], metrics=metrics)

# List × list → ResultsCollection
results = crossref.compare([doc_a, doc_b], [doc_c, doc_d], metrics=metrics)

# Parallel pair scoring (uses ThreadPoolExecutor; safe with metric-level workers)
results = crossref.compare([doc_a, doc_b, doc_c], metrics=metrics, workers=4)
```

**Note on `workers`:** `compare(workers=N)` parallelizes across document pairs using threads. Several metrics also accept their own `workers` parameter for row-level process parallelism. When using both, total spawned processes = `compare_workers × metric_workers`. For large runs, keep one of them at 1.

---

## Preprocessing

Preprocessing is applied to passage text before scoring. The original `Passage.text` is never modified.

```python
from crossref.preprocessing import generate_preprocessing_pipeline

pipeline = generate_preprocessing_pipeline([
    {"replace_punctuation": [["'", "'"], ["—", " "]]},
    {"normalize_text": True},
    {"remove_extra_spaces": True},
    {"remove_stopwords": ["yea", "behold", "and it came to pass"]},
])

results = crossref.compare(doc_a, doc_b, metrics=metrics, preprocess_fn=pipeline)
```

Or pass a plain function:

```python
results = crossref.compare(doc_a, doc_b, metrics=metrics, preprocess_fn=str.lower)
```

### Available preprocessing steps

| Step | Description |
|---|---|
| `normalize_text` | Lowercase, normalize Unicode |
| `replace_punctuation` | Replace specific characters: `[["'", "'"]]` |
| `remove_all_punctuation_except` | Keep only specified characters |
| `remove_punctuation` | Remove all punctuation |
| `remove_stopwords` | NLTK English stopwords + optional custom phrases |
| `remove_extra_spaces` | Collapse whitespace |
| `remove_diacritics` | Strip accent marks (é→e) |
| `normalize_unicode` | NFKC normalization |
| `remove_numbers` | Strip digit sequences |
| `remove_short_words` | Drop words shorter than N characters |
| `stem` | Porter stemming (NLTK) |
| `lemmatize` | WordNet lemmatization (NLTK) |

---

## Metrics

All metrics implement `score(text1, text2) → float` and `score_all(texts1, texts2) → np.ndarray`. Scores are in `[0, 1]` where 1.0 = identical.

### Syntactic

| Metric | Name | Description | Extra deps |
|---|---|---|---|
| `NGramMetric` | `"ngram"` | Largest non-overlapping common n-grams, normalized by self-similarity | — |
| `JaccardMetric` | `"jaccard"` | Word set intersection / union | — |
| `CharNGramMetric` | `"charngram"` | Character n-gram Jaccard | — |
| `LCSMetric` | `"lcs"` | Longest common subsequence / min(len1, len2) | — |
| `LevenshteinMetric` | `"levenshtein"` | 1 − word edit distance / max(len1, len2) | — |
| `LCSubstringMetric` | `"lcsubstring"` | Longest contiguous common substring | — |
| `DiceMetric` | `"dice"` | Dice coefficient on word sets | — |
| `ROUGELMetric` | `"rouge_l"` | ROUGE-L F1 (LCS-based) | — |
| `ROUGENMetric` | `"rouge_n"` | ROUGE-N F1 (n-gram count overlap) | — |
| `FuzzyNGramMetric` | `"fuzzy_ngram"` | rapidfuzz token_set_ratio | `[fuzzy]` |
| `PhoneticMetric` | `"phonetic"` | Soundex/Metaphone/NYSIIS Jaccard | `[phonetic]` |

### Semantic

| Metric | Name | Description | Extra deps |
|---|---|---|---|
| `SentenceTransformerMetric` | `"sentencetransformer"` | Cosine similarity of sentence embeddings | — |
| `CrossEncoderMetric` | `"crossencoder"` | Cross-encoder reranking score | — |
| `ColBERTMetric` | `"colbert"` | Late-interaction ColBERT scoring | `[colbert]` |
| `LLMJudgeMetric` | `"llm_judge"` | LLM-based similarity judgment | `[llm]` |
| `APIEmbeddingMetric` | `"api_embedding"` | Embedding via API (OpenAI, Cohere, Voyage) | `[llm]` |
| `Doc2VecMetric` | `"doc2vec"` | Gensim Doc2Vec embeddings | `[doc2vec]` |
| `AveragedWordVecMetric` | `"avgwordvec"` | Averaged GloVe word vectors | `[doc2vec]` |
| `WMDMetric` | `"wmd"` | Word Mover's Distance | `[doc2vec]` |
| `SoftCosineMetric` | `"soft_cosine"` | Soft cosine via word vector similarity | `[doc2vec]` |

### Statistical

| Metric | Name | Description | Extra deps |
|---|---|---|---|
| `NCDMetric` | `"ncd"` | Normalized Compression Distance (zlib) | — |
| `TFIDFMetric` | `"tfidf"` | TF-IDF cosine similarity | `[tfidf]` |
| `BM25Metric` | `"bm25"` | BM25 retrieval score | `[bm25]` |
| `LSAMetric` | `"lsa"` | Latent Semantic Analysis cosine | `[tfidf]` |
| `JSMetric` | `"js"` | Jensen-Shannon divergence on word distributions | — |
| `LZJDMetric` | `"lzjd"` | Lempel-Ziv Jaccard Distance | — |

### Recommended starting point

```python
metrics = [
    crossref.NGramMetric(n_min=2, n_max=10),                       # fast, interpretable
    crossref.SentenceTransformerMetric("all-MiniLM-L6-v2"),        # fastest semantic
    # crossref.SentenceTransformerMetric("all-mpnet-base-v2"),     # best quality semantic
]
```

---

## Results API

### `top_k`

```python
# By passage index
matches = results.top_k(42, metric="ngram", k=10)
# → list of Match(score, index, label, text)

# By label string (exact match, then substring fallback)
matches = results.top_k("1 Nephi 3:7", metric="ngram", k=10)

# Return all passages ranked (k=None)
matches = results.top_k(0, metric="ngram", k=None)
```

### `filter`

```python
matches = results.filter(metric="ngram", threshold=0.5)
# → list of FilterMatch(score, index1, index2, label1, label2, text1, text2)

# Memory-efficient for HDF5-backed results (reads one row at a time)
```

### `summary`

```python
stats = results.summary(metric="ngram")
# → {"mean": 0.12, "median": 0.04, "std": 0.18,
#    "min": 0.0, "max": 1.0, "shape": (6604, 6604),
#    "above_0.5": 412, "above_0.8": 88, "above_0.9": 31}
```

### `to_csv`

```python
results.to_csv("output.csv", metric="ngram", threshold=0.3)
```

### `heatmap`

```python
results.heatmap(metric="ngram", output="heatmap.png")
results.heatmap(metric="sentencetransformer", output="sem.png", xlabels=True, ylabels=True)
```

### `diff`

Compare two Results objects element-wise for all shared metrics (e.g. BoM-KJV minus BoM-JST):

```python
delta = results_bom_kjv.diff(results_bom_jst)
# → Results with matrices = kjv_matrix - jst_matrix for each shared metric
```

### `explain`

```python
ev = results.explain(42, 17, metric="ngram")
# → {"score": 0.72, "common_ngrams": {4: {("and", "it", "came", "to"), ...}}}
```

---

## ResultsCollection

When comparing multiple documents, `compare()` returns a `ResultsCollection`.

```python
collection = crossref.compare([bom, kjv, dc], metrics=metrics)

# Index by (docs1_index, docs2_index)
r = collection[0, 1]   # bom vs kjv → Results

# Iterate
for (i, j), results in collection.items():
    print(results.doc1_title, "vs", results.doc2_title)

# Named pairs
for title1, title2, results in collection.pairs():
    results.top_k(0, metric="ngram", k=5)

# Filter across all pairs
matches = collection.filter(metric="ngram", threshold=0.6)
# → list of CollectionMatch(score, index1, index2, label1, label2,
#                           text1, text2, doc1_title, doc2_title)
```

---

## Saving and Loading

Backend is inferred from the file extension.

```python
# Pickle (default, loads full matrices into RAM)
results.save("results.pkl")

# HDF5 (compressed, lazy row reads — best for large runs)
results.save("results.h5")           # requires pip install "crossref[hdf5]"

# SQLite (sparse — stores only scores above threshold, fast queries)
results.save("results.db", threshold=0.3)

# Load (backend auto-detected)
results = crossref.load("results.pkl")
results = crossref.load("results.h5")
results = crossref.load("results.db")

# Or directly
results = crossref.Results.load("results.pkl")
collection = crossref.ResultsCollection.load("collection.h5")
```

### Choosing a backend

| Backend | Best for | Notes |
|---|---|---|
| `.pkl` | Small to medium runs | Loads full matrix into RAM |
| `.h5` | Large runs (millions of pairs) | Gzip-compressed; `filter()` reads one row at a time |
| `.db` | Very large runs, sparse results | Only stores pairs above threshold; fast SQL queries |

---

## YAML Configuration (CLI)

```yaml
# Run with: crossref config.yaml --output results.h5

workers: 4   # parallel document-pair scoring threads (default: 1)

# All-vs-all (N×N ResultsCollection):
documents:
  - pathname: ~/texts/book_of_mormon.md
    splitter: line

  - pathname: ~/texts/kjv_new_testament.md
    splitter: line

# Or list × list (M×N ResultsCollection, avoids spurious self-pairs):
# documents1:
#   - pathname: ~/texts/genesis/
#     splitter: line
# documents2:
#   - pathname: ~/texts/book_of_mormon.md
#     splitter: line
#   - pathname: ~/texts/kjv_new_testament.md
#     splitter: line

preprocessing:
  - replace_punctuation:
      - ["'", "'"]    # curly apostrophe → straight
      - ["—", " "]    # em-dash → space
  - normalize_text: true
  - remove_extra_spaces: true
  - remove_stopwords:
      - "yea"
      - "behold"
      - "and it came to pass"

metrics:
  - name: ngram
    n_min: 2
    n_max: 10
    custom_stopphrases:
      - "thus saith the lord"

  - name: sentencetransformer
    modelname: all-MiniLM-L6-v2
    # modelname: all-mpnet-base-v2

  # Additional metrics (see full list above):
  # - name: jaccard
  # - name: lcs
  # - name: tfidf
  # - name: bm25
  #   k1: 1.5
  #   b: 0.75
```

---

## Custom Metrics

Subclass `SimilarityMetric` and implement `name`, `score()`, and optionally `score_all()`:

```python
from crossref.metrics.base import SimilarityMetric
import numpy as np

class MyMetric(SimilarityMetric):
    name = "my_metric"

    def score(self, text1: str, text2: str) -> float:
        # Your similarity logic here
        ...

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        # Optional: override for batched efficiency
        ...

    def config(self) -> dict:
        return {"name": self.name}

results = crossref.compare(doc_a, doc_b, metrics=[MyMetric()])
```

---

## Example: Book of Mormon / KJV Comparison

```python
import crossref

bom = crossref.Document.from_file("~/texts/book_of_mormon.md", splitter="line")
kjv = crossref.Document.from_file("~/texts/kjv_bible.md", splitter="line")

pipeline = crossref.generate_preprocessing_pipeline([
    {"replace_punctuation": [["'", "'"]]},
    {"normalize_text": True},
    {"remove_stopwords": ["yea", "behold", "and it came to pass"]},
])

metrics = [
    crossref.NGramMetric(n_min=2, n_max=10, workers=8),
    crossref.SentenceTransformerMetric("all-mpnet-base-v2"),
]

results = crossref.compare(bom, kjv, metrics=metrics, preprocess_fn=pipeline)
results.save("bom_kjv.h5")   # compressed, lazy-loading HDF5

# Find the most similar passage pairs
for match in results.filter(metric="ngram", threshold=0.6):
    print(f"[{match.score:.2f}] {match.label1}  →  {match.label2}")
    print(f"  {match.text1}")
    print(f"  {match.text2}")
```

---

## License

MIT
