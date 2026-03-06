import argparse
import os
import yaml

from .documents import Document
from .preprocessing import generate_preprocessing_pipeline
from .compare import compare


def main():
    args = parse_args()
    config = load_config(args.config)
    preprocess_fn = generate_preprocessing_pipeline(config)
    documents = load_documents(config, preprocess_fn)
    metrics = load_metrics(config)

    results = compare(documents, metrics=metrics)

    print(f"Compared {len(documents)} document(s)")
    for doc in documents:
        print(f"  {doc.title}: {len(doc)} passages")

    if args.output:
        results.save(args.output)
        print(f"Results saved to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrossRef: Document Cross-Referencing Tool")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save results (pickle). If not provided, results are not saved.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_documents(config: dict, preprocess_fn) -> list[Document]:
    documents = []
    for doc_config in config['documents']:
        document = Document.from_file(
            path=os.path.expanduser(doc_config['pathname']),
            splitter=doc_config.get('splitter', 'line'),
            preprocess_fn=preprocess_fn,
            title=doc_config.get('title'),
        )
        documents.append(document)
    return documents


def load_metrics(config: dict) -> list[SimilarityMetric]:
    from .metrics import metric_from_config
    return [metric_from_config(mc) for mc in config['metrics']]


if __name__ == "__main__":
    main()
