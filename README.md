# CrossRef
Python package for automated document cross-referencing


# Ideas:
*   N-Gram Similarity: (for word & phrase matching)
    *   https://github.com/seatgeek/thefuzz
    *   https://spacy.io/usage/linguistic-features#vectors-similarity
    *   Could I combine n-grams with word vectors to get a ngram-vector?
    *   RapidFuzz
*   Paragraph Embeddings & Cosine Similarity (for thematic / semantic matching)
    *   Doc2Vec
*   Clean input text
    *   Normalize spelling
    *   Remove special characters appropriately
*   Data Mining Techniques:
    *   Minhash / LSH

*   I should be able to subtract the difference between two comparisons
    *   Use Case: Compare BoM-KJV, compare BoM-JST, & then find the difference
        to see how the BoM correlates to the changes the JST made to the KJV