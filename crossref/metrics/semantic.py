import numpy as np
import torch

# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, SimilarityFunction

from crossref.metrics.base import SimilarityMetric


class SemanticSimilarityMetric(SimilarityMetric):
    def __init__(self):
        pass


class SentenceTransformerMetric(SemanticSimilarityMetric):
    """
        https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

        Best model: all-mpnet-base-v2
        Fastest model w/ close performance: all-MiniLM-L6-v2
    """
    def __init__(self, modelname: str, similarity_fn: SimilarityFunction = SimilarityFunction.COSINE):
        super().__init__()
        self.modelname = modelname
        self.similarity_fn = similarity_fn
        self.model = SentenceTransformer(modelname, similarity_fn_name=similarity_fn)

    def score(self, text1: str, text2: str) -> float:
        similarity: torch.Tensor = self._similarity(text1, text2)
        return similarity.item()

    def score_all(self, texts1: list[str], texts2: list[str]) -> np.ndarray:
        similarities: torch.Tensor = self._similarity(texts1, texts2)
        return similarities.numpy()

    def _similarity(self, texts1: str | list[str], texts2: str | list[str]) -> 'torch.Tensor':
        embeddings1: np.ndarray = self.model.encode(texts1)
        embeddings2: np.ndarray = self.model.encode(texts2)
        return self.model.similarity(embeddings1, embeddings2)


class Doc2VecMetric(SemanticSimilarityMetric):
    """
    https://radimrehurek.com/gensim/models/doc2vec.html
    https://spotintelligence.com/2023/09/06/doc2vec/
    """
    def __init__(self):
        super().init()

    def score(self, text1: str, text2: str) -> float:
        raise NotImplementedError()  # TODO


# def main():
#     metric = SentenceTransformerMetric(modelname='all-mpnet-base-v2')
#     texts: list[str] = [
#         "For it is expedient that there should be a great and last sacrifice; yea, not a sacrifice of man, neither of beast, neither of any manner of fowl; for it shall not be a human sacrifice; but it must be an infinite and eternal sacrifice.",
#         "And behold, this is the whole meaning of the law, every whit pointing to that great and last sacrifice; and that great and last sacrifice will be the Son of God, yea, infinite and eternal.",
#         "But behold, the Spirit hath said this much unto me, saying: Cry unto this people, saying Repent ye, and prepare the way of the Lord, and walk in his paths, which are straight; for behold, the kingdom of heaven is at hand, and the Son of God cometh upon the face of the earth.",
#         "And saying, Repent ye: for the kingdom of heaven is at hand.",
#         "For this is he that was spoken of by the prophet Esaias, saying, The voice of one crying in the wilderness, Prepare ye the way of the Lord, make his paths straight.",
#     ]

#     import time
#     start = time.time()
#     score = metric.score_multiple(texts, texts)
#     print("Time Elapsed: ", time.time() - start)
#     print("Score:\n", score)


# if __name__ == "__main__":
#     main()
