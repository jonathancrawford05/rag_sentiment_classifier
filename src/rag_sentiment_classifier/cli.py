import argparse
import json
import logging

from rag_sentiment_classifier.config.settings import get_settings
from rag_sentiment_classifier.models.document import DocumentInput
from rag_sentiment_classifier.services.classification_service import (
    DocumentClassificationService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a document with Ollama.")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--metadata", help="JSON metadata", default=None)
    return parser


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    args = build_parser().parse_args()

    metadata = json.loads(args.metadata) if args.metadata else None
    document = DocumentInput(
        content=args.content,
        document_id=args.document_id,
        source=args.source,
        metadata=metadata,
    )

    service = DocumentClassificationService()
    result = service.classify_document(document)
    if result is None:
        raise SystemExit("Classification failed")

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
