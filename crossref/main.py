import argparse
import logging
import os
import yaml

from .documents import Document
from .preprocessing import generate_preprocessing_pipeline
from .compare import compare

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    logger.debug("Loading config: %s", args.config)
    config = load_config(args.config)
    preprocess_fn = generate_preprocessing_pipeline(config)
    docs1, docs2 = load_documents(config)
    metrics = load_metrics(config)
    logger.info(
        "Loaded %d metric(s): %s",
        len(metrics),
        ", ".join(m.name for m in metrics),
    )

    results = compare(
        docs1,
        docs2,
        metrics=metrics,
        preprocess_fn=preprocess_fn,
        preprocessing_config=config.get('preprocessing'),
        workers=config.get('workers', 1),
    )

    if docs2 is None:
        all_docs = docs1
    else:
        seen, all_docs = set(), []
        for d in docs1 + docs2:
            if id(d) not in seen:
                seen.add(id(d))
                all_docs.append(d)
    logger.info("Comparison complete — %d document(s):", len(all_docs))
    for doc in all_docs:
        logger.info("  %s: %d passages", doc.title, len(doc))

    if args.output:
        results.save(args.output)
        logger.info("Results saved to %s", args.output)


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


def load_documents(config: dict) -> tuple[list[Document], list[Document] | None]:
    """Return (docs1, docs2).

    If the config has a top-level 'documents' key, docs2 is None (all-vs-all or
    single-pair depending on count). If it has 'documents1'/'documents2' keys,
    docs2 is a separate list (list×list topology).
    """
    if 'documents1' in config:
        if 'documents2' not in config:
            raise ValueError("Config has 'documents1' but no 'documents2'.")
        return (
            [_load_document(dc) for dc in config['documents1']],
            [_load_document(dc) for dc in config['documents2']],
        )
    return [_load_document(dc) for dc in config['documents']], None


def _load_document(doc_config: dict) -> Document:
    path = os.path.expanduser(doc_config['pathname'])
    title = doc_config.get('title')
    # Format is inferred from extension; 'format' key overrides if provided.
    fmt = doc_config.get('format', os.path.splitext(path)[1].lstrip('.').lower())

    if fmt in ('csv', 'tsv'):
        return Document.from_csv(
            path,
            text_column=doc_config['text_column'],
            label_column=doc_config.get('label_column'),
            delimiter=doc_config.get('delimiter'),
            title=title,
        )
    if fmt == 'json':
        return Document.from_json(
            path,
            text_key=doc_config['text_key'],
            label_key=doc_config.get('label_key'),
            title=title,
        )
    if fmt in ('jsonl', 'ndjson'):
        return Document.from_jsonl(
            path,
            text_key=doc_config['text_key'],
            label_key=doc_config.get('label_key'),
            title=title,
        )
    return Document.from_file(
        path,
        splitter=doc_config.get('splitter', 'line'),
        title=title,
    )


def load_metrics(config: dict) -> list[SimilarityMetric]:
    from .metrics import metric_from_config
    return [metric_from_config(mc) for mc in config['metrics']]


if __name__ == "__main__":
    main()
