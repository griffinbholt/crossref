from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string
import sys

from comparison import ComparisonScorer


class WordComparisonScorer(ComparisonScorer):
    def __init__(self):
        pass

class NGramScorer(WordComparisonScorer):
    def __init__(self, n_min: int, n_max: int, custom_stopphrases: list[str]):
        super().__init__()
        self.n_min: int = n_min
        self.n_max: int = n_max

        self.stopgrams: dict[int, set[tuple[str, ...]]] = {n: set() for n in range(n_min, n_max + 1)}
        for stopgram in (stopwords.words('english') + custom_stopphrases):
            stopgram_tokenized = tuple(word_tokenize(stopgram))
            self.stopgrams[len(stopgram_tokenized)].add(stopgram_tokenized)

    def score(self, text1: str, text2: str, return_full: bool = False) -> float:
        common_ngrams: dict[int, set[tuple[str, ...]]] = self._find_common_ngrams(text1, text2)
        score: float = sum(n * len(phrases) for n, phrases in common_ngrams.items())
        if return_full:
            return score, common_ngrams
        return score

    def _find_common_ngrams(
            self,
            text1: str,
            text2: str
        ) -> dict[int, set[tuple[str, ...]]]:
        """
        Find common n-grams between two paragraphs using NLTK.
        Returns only the largest n-grams (doesn't include sub-n-grams).
        """
        # Generate ngrams for both texts
        ngrams_text1: dict[int, set[tuple[str, ...]]] = self._generate_ngrams_nltk(text1)
        ngrams_text2: dict[int, set[tuple[str, ...]]] = self._generate_ngrams_nltk(text2)

        # Find common n-grams for each value of n
        common_ngrams = {n: ngrams_text1[n] & ngrams_text2[n] for n in range(self.n_min, self.n_max + 1)}
        
        # Remove sub-n-grams that are part of larger common n-grams
        filtered_ngrams: dict[int, set[tuple[str, ...]]] = {}
        for n in range(self.n_max, self.n_min - 1, -1):  # Start with the largest n-grams
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

    def _generate_ngrams_nltk(
            self,
            text: str
        ) -> dict[int, set[tuple[str, ...]]]:
        """
        Generate n-grams from a text for all n from n_min to n_max using NLTK.
        Returns a dictionary with n as keys and sets of n-grams as values.
        """
        cleaned_text: str = text.strip().lower().translate(str.maketrans('', '', string.punctuation))  # TODO: Maybe clean text before hand
        tokens: list[str] = word_tokenize(cleaned_text)
        text_ngrams: dict[int, set[tuple[str, ...]]] =  {n: set(ngrams(tokens, n)) - self.stopgrams[n] for n in range(self.n_min, self.n_max + 1)}
        return text_ngrams

class FuzzyNGramScorer(WordComparisonScorer):
    def __init__(self):
        super().__init__()
    
    def score(self, text1: str, text2: str) -> float:
        pass


def main(i: int, j: int):
    custom_stopphrases = [
        'ye',
        'yea',
        'and it came to pass',
    ]  # TODO: You could automate custom stopphrases by removing the most common ngrams from the document already
    ngram_scorer = NGramScorer(n_min=1, n_max=10, custom_stopphrases=custom_stopphrases)
    texts: list[str] = [
        "For it is expedient that there should be a great and last sacrifice; yea, not a sacrifice of man, neither of beast, neither of any manner of fowl; for it shall not be a human sacrifice; but it must be an infinite and eternal sacrifice.",
        "And behold, this is the whole meaning of the law, every whit pointing to that great and last sacrifice; and that great and last sacrifice will be the Son of God, yea, infinite and eternal.",
        "But behold, the Spirit hath said this much unto me, saying: Cry unto this people, saying Repent ye, and prepare the way of the Lord, and walk in his paths, which are straight; for behold, the kingdom of heaven is at hand, and the Son of God cometh upon the face of the earth.",
        "And saying, Repent ye: for the kingdom of heaven is at hand.",
        "For this is he that was spoken of by the prophet Esaias, saying, The voice of one crying in the wilderness, Prepare ye the way of the Lord, make his paths straight.",
    ]

    import time
    start = time.time()
    score, ngrams = ngram_scorer.score(texts[i], texts[j], return_full=True)
    print("Time Elapsed: ", time.time() - start)
    print("Score: ", score)
    print("N-Grams:", ngrams)

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
