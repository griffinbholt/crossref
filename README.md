# crossref
Python package for automated document cross-referencing


# Ideas:
*   N-Gram Similarity: (for word & phrase matching)
    *   https://github.com/seatgeek/thefuzz
    *   https://spacy.io/usage/linguistic-features#vectors-similarity
    *   Could I combine n-grams with word vectors to get a ngram-vector?
    *   RapidFuzz
*   Paragraph Embeddings & Cosine Similarity (for thematic / semantic matching)
    *   SBERT: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
*   Clean input text
    *   Normalize spelling
    *   Remove special characters appropriately