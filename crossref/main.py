import argparse
import os
import yaml

from parsing import Document, InputFormat
from metrics.base import SimilarityMetric


def main():
    args: argparse.Namespace = parse_args()
    config: dict = load_config(args.config)
    documents: list[Document] = load_documents(config)
    metrics: list[SimilarityMetric] = load_metrics(config)
    # TODO - Clean the sources
    # TODO - Start the various subprocess (in a queue for parallel processing?)
    # TODO - Compile the reports
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrossRef")
    parser.add_argument("config", type=str, help="Path to the config file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_documents(config: dict) -> list[Document]:
    documents: list[Document] = []
    for doc_config in config['documents']:
        document = Document(
            path=os.path.expanduser(doc_config['path']),
            format=InputFormat[doc_config['format']]
        )
        documents.append(document)
    return documents


def load_metrics(config: dict) -> list[SimilarityMetric]:
    metrics: list[SimilarityMetric] = []
    for metric_config in config['metrics']:
        metric_type: str = metric_config['type']

        # Late imports to prevent unnecessary imports
        if metric_type == 'sentencetransformer':
            from metrics.semantic import SentenceTransformerMetric
            metrics.append(SentenceTransformerMetric(metric_config['modelname']))
        elif metric_type == 'ngram':
            from metrics.syntactic import NGramMetric
            metrics.append(NGramMetric(metric_config['n_min'], metric_config['n_max'], metric_config['custom_stopphrases']))
        else:
            raise NotImplementedError(f"Unsupported metric type: {metric_type}")

    return metrics


if __name__ == "__main__":
    main()