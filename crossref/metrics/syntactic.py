import numpy as np

from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from base import SimilarityMetric
from crossref.preprocessing import remove_punctuation, normalize_text, remove_stopwords

class SyntacticSimilarityMetric(SimilarityMetric):
    def __init__(self):
        pass


class NGramMetric(SyntacticSimilarityMetric):
    def __init__(self, n_min: int, custom_stopphrases: set[str]):
        super().__init__()
        self.n_min: int = n_min
        self.custom_stopphrases: set[str] = custom_stopphrases

    def score(self, text1: str, text2: str, return_full: bool = False) -> float | tuple[float, dict[int, set[tuple[str, ...]]]]:
        ngrams_text1: dict[int, set[tuple[str, ...]]] = self._generate_ngrams(text1)
        ngrams_text2: dict[int, set[tuple[str, ...]]] = self._generate_ngrams(text2)        
        common_ngrams: dict[int, set[tuple[str, ...]]] = self._find_common_largest_ngrams(ngrams_text1, ngrams_text2)
        score: float = sum(n * len(phrases) for n, phrases in common_ngrams.items())
        if return_full:
            return score, common_ngrams
        return score

    def score_all(self, texts1: list[str], texts2: list[str], return_full: bool = False) -> np.ndarray | tuple[float | list[list[dict[int, set[tuple[str, ...]]]]]]:
        ngram_cache1: list[dict[int, set[tuple[str, ...]]]] = [self._generate_ngrams(text) for text in texts1]
        ngram_cache2: list[dict[int, set[tuple[str, ...]]]] = [self._generate_ngrams(text) for text in texts2]

        # TODO: Could be parallelized
        complete_common_ngrams: list[list[dict[int, set[tuple[str, ...]]]]] = [[None] * len(texts2) for _ in range(len(texts1))]
        scores = np.zeros((len(texts1), len(texts2)))
        for i in range(len(texts1)):
            ngrams_text1: dict[int, set[tuple[str, ...]]] = ngram_cache1[i]
            for j in range(len(texts2)):
                if i == j:
                    n_tokens = max(ngrams_text1.keys())
                    scores[i, j] = float(n_tokens)
                    if return_full:
                        complete_common_ngrams[i][j] = {n_tokens: ngrams_text1[n_tokens]}
                    continue

                ngrams_text2: dict[int, set[tuple[str, ...]]] = ngram_cache2[j]
                common_ngrams = self._find_common_largest_ngrams(ngrams_text1, ngrams_text2)
                scores[i, j] = self._compute_ngram_score(common_ngrams)
                if return_full:
                    complete_common_ngrams[i][j] = common_ngrams

        # Normalize the scores
        diag_scores = np.diagonal(scores)
        nmlzd_scores = scores / np.minimum(diag_scores[:, np.newaxis], diag_scores[np.newaxis, :])

        if return_full:
            return nmlzd_scores, complete_common_ngrams
        return nmlzd_scores

    def _generate_ngrams(
            self,
            text: str
        ) -> dict[int, set[tuple[str, ...]]]:
        """
        Generate n-grams from a text for all n from n_min to n_max using NLTK.
        Returns a dictionary with n as keys and sets of n-grams as values.
        """
        cleaned_text: str = remove_stopwords(normalize_text(remove_punctuation(text)), self.custom_stopphrases)
        tokens: list[str] = word_tokenize(cleaned_text)
        text_ngrams: dict[int, set[tuple[str, ...]]] =  {n: set(ngrams(tokens, n)) for n in range(self.n_min, len(tokens) + 1)}
        return text_ngrams

    def _find_common_largest_ngrams(
            self,
            ngrams_text1: dict[int, set[tuple[str, ...]]],
            ngrams_text2: dict[int, set[tuple[str, ...]]]
        ) -> dict[int, set[tuple[str, ...]]]:
        """
        Find common n-grams between two paragraphs using NLTK.
        Returns only the largest n-grams (doesn't include sub-n-grams).
        """
        # Find common n-grams for each value of n
        n_max: int = min(max(ngrams_text1.keys()), max(ngrams_text2.keys()))
        common_ngrams = {n: ngrams_text1[n] & ngrams_text2[n] for n in range(self.n_min, n_max + 1)}
        
        # Remove sub-n-grams that are part of larger common n-grams
        filtered_ngrams: dict[int, set[tuple[str, ...]]] = {}
        for n in range(n_max, self.n_min - 1, -1):  # Start with the largest n-grams
            for ngram in common_ngrams[n]:
                # Add this n-gram to the filtered set
                if n not in filtered_ngrams:
                    filtered_ngrams[n] = set()
                filtered_ngrams[n].add(ngram)
                # Remove all sub-n-grams of this n-gram from consideration
                if n > self.n_min:
                    for sub_n in range(n - 1, self.n_min - 1, -1):
                        for i in range(n - sub_n + 1):
                            sub_ngram = ngram[i:i+sub_n]
                            if sub_ngram in common_ngrams[sub_n]:
                                common_ngrams[sub_n].remove(sub_ngram)

        return filtered_ngrams

    def _compute_ngram_score(self, common_ngrams: dict[int, set[tuple[str, ...]]]) -> float:
        return float(sum(n * len(phrases) for n, phrases in common_ngrams.items()))


class FuzzyNGramMetric(SyntacticSimilarityMetric):
    def __init__(self):
        super().__init__()
    
    def score(self, text1: str, text2: str) -> float:
        raise NotImplementedError()  # TODO


# def main(i: int = None, j: int = None):
    # custom_stopphrases = {
    #     'ye',
    #     'yea',
    #     'and it came to pass',
    #     'behold'
    # }  # TODO: You could automate custom stopphrases by removing the most common ngrams from the document already
    # ngram_scorer = NGramMetric(n_min=2, custom_stopphrases=custom_stopphrases)
    # texts: list[str] = [
    #     "For it is expedient that there should be a great and last sacrifice; yea, not a sacrifice of man, neither of beast, neither of any manner of fowl; for it shall not be a human sacrifice; but it must be an infinite and eternal sacrifice.",
    #     "And behold, this is the whole meaning of the law, every whit pointing to that great and last sacrifice; and that great and last sacrifice will be the Son of God, yea, infinite and eternal.",
    #     "But behold, the Spirit hath said this much unto me, saying: Cry unto this people, saying Repent ye, and prepare the way of the Lord, and walk in his paths, which are straight; for behold, the kingdom of heaven is at hand, and the Son of God cometh upon the face of the earth.",
    #     "And saying, Repent ye: for the kingdom of heaven is at hand.",
    #     "For this is he that was spoken of by the prophet Esaias, saying, The voice of one crying in the wilderness, Prepare ye the way of the Lord, make his paths straight.",
    # ]

    # import time
    # start = time.time()
    # scores, common_ngrams = ngram_scorer.score_all(texts, texts, return_full=True)
    # print("Time Elapsed: ", time.time() - start)
    # print("Scores:\n", scores)
    # for i in range(len(texts)):
    #     for j in range(len(texts)):
    #         print(f"Common N-Grams {i}, {j}:", common_ngrams[i][j])


# if __name__ == "__main__":
#     main()
