import pytest
from pydantic import ValidationError

from rag_sentiment_classifier.models.document import ClassificationResult, DocumentInput


def test_document_input_strips_content() -> None:
    doc = DocumentInput(
        content="  Test content  ",
        document_id="DOC-001",
        source="unit",
    )
    assert doc.content == "Test content"


def test_document_input_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        DocumentInput(content="   ", document_id="DOC-001", source="unit")


def test_classification_result_validation() -> None:
    result = ClassificationResult(
        document_id="DOC-001",
        category="Regulatory",
        confidence=0.9,
        subcategories=["SEC"],
        risk_level="high",
        requires_review=False,
        reasoning="Clear regulatory filing.",
    )
    assert result.category == "Regulatory"
