"""
Unit tests for data models.

Tests Pydantic models including:
- DocumentInput validation and sanitization
- ClassificationResult validation
- Field constraints and patterns
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_sentiment_classifier.models.document import ClassificationResult, DocumentInput


class TestDocumentInputValidation:
    """Test DocumentInput validation and sanitization."""

    def test_document_input_valid(self) -> None:
        """Test creating valid DocumentInput."""
        doc = DocumentInput(
            content="Test content",
            document_id="DOC-001",
            source="test",
        )

        assert doc.content == "Test content"
        assert doc.document_id == "DOC-001"
        assert doc.source == "test"
        assert doc.metadata is None

    def test_document_input_strips_content(self) -> None:
        """Test content is stripped of leading/trailing whitespace."""
        doc = DocumentInput(
            content="  Test content  ",
            document_id="DOC-001",
            source="test",
        )

        assert doc.content == "Test content"

    def test_document_input_rejects_empty_content(self) -> None:
        """Test empty content after stripping is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentInput(content="   ", document_id="DOC-001", source="test")

        assert "Content cannot be empty" in str(exc_info.value)

    def test_document_input_with_metadata(self) -> None:
        """Test DocumentInput with valid metadata."""
        doc = DocumentInput(
            content="Test content",
            document_id="DOC-001",
            source="test",
            metadata={"key1": "value1", "key2": "value2"},
        )

        assert doc.metadata == {"key1": "value1", "key2": "value2"}

    def test_document_input_content_sanitization(self) -> None:
        """Test control characters are removed from content."""
        # Content with null bytes and other control chars
        doc = DocumentInput(
            content="Test\x00content\x01with\x02control\x03chars",
            document_id="DOC-001",
            source="test",
        )

        # Control characters should be removed
        assert "\x00" not in doc.content
        assert "\x01" not in doc.content
        assert "Testcontentwithcontrolchars" == doc.content

    def test_document_input_preserves_newlines_tabs(self) -> None:
        """Test that newlines and tabs are preserved."""
        doc = DocumentInput(
            content="Line 1\nLine 2\tTabbed",
            document_id="DOC-001",
            source="test",
        )

        assert "\n" in doc.content
        assert "\t" in doc.content
        assert doc.content == "Line 1\nLine 2\tTabbed"


class TestDocumentIDValidation:
    """Test document ID pattern validation."""

    def test_valid_document_id_alphanumeric(self) -> None:
        """Test valid alphanumeric document ID."""
        doc = DocumentInput(
            content="Test",
            document_id="DOC001",
            source="test",
        )

        assert doc.document_id == "DOC001"

    def test_valid_document_id_with_dash(self) -> None:
        """Test valid document ID with dashes."""
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001-TEST",
            source="test",
        )

        assert doc.document_id == "DOC-001-TEST"

    def test_valid_document_id_with_underscore(self) -> None:
        """Test valid document ID with underscores."""
        doc = DocumentInput(
            content="Test",
            document_id="DOC_001_TEST",
            source="test",
        )

        assert doc.document_id == "DOC_001_TEST"

    def test_invalid_document_id_with_space(self) -> None:
        """Test document ID with spaces is rejected."""
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id="DOC 001",
                source="test",
            )

    def test_invalid_document_id_with_special_chars(self) -> None:
        """Test document ID with special characters is rejected."""
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id="DOC@#$%",
                source="test",
            )

    def test_document_id_max_length(self) -> None:
        """Test document ID can be up to 100 characters."""
        long_id = "D" * 100
        doc = DocumentInput(
            content="Test",
            document_id=long_id,
            source="test",
        )

        assert len(doc.document_id) == 100

    def test_document_id_too_long(self) -> None:
        """Test document ID exceeding 100 characters is rejected."""
        too_long_id = "D" * 101
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id=too_long_id,
                source="test",
            )

    def test_document_id_empty(self) -> None:
        """Test empty document ID is rejected."""
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id="",
                source="test",
            )


class TestSourceValidation:
    """Test source field validation."""

    def test_valid_source(self) -> None:
        """Test valid source field."""
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001",
            source="test-source",
        )

        assert doc.source == "test-source"

    def test_source_empty_rejected(self) -> None:
        """Test empty source is rejected."""
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id="DOC-001",
                source="",
            )

    def test_source_max_length(self) -> None:
        """Test source can be up to 100 characters."""
        long_source = "S" * 100
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001",
            source=long_source,
        )

        assert len(doc.source) == 100

    def test_source_too_long(self) -> None:
        """Test source exceeding 100 characters is rejected."""
        too_long_source = "S" * 101
        with pytest.raises(ValidationError):
            DocumentInput(
                content="Test",
                document_id="DOC-001",
                source=too_long_source,
            )


class TestContentValidation:
    """Test content field validation."""

    def test_content_min_length(self) -> None:
        """Test content must be at least 1 character."""
        doc = DocumentInput(
            content="A",
            document_id="DOC-001",
            source="test",
        )

        assert doc.content == "A"

    def test_content_max_length(self) -> None:
        """Test content can be up to 50,000 characters."""
        long_content = "A" * 50000
        doc = DocumentInput(
            content=long_content,
            document_id="DOC-001",
            source="test",
        )

        assert len(doc.content) == 50000

    def test_content_too_long(self) -> None:
        """Test content exceeding 50,000 characters is rejected."""
        too_long_content = "A" * 50001
        with pytest.raises(ValidationError):
            DocumentInput(
                content=too_long_content,
                document_id="DOC-001",
                source="test",
            )


class TestMetadataValidation:
    """Test metadata field validation."""

    def test_metadata_valid(self) -> None:
        """Test valid metadata."""
        metadata = {"key1": "value1", "key2": 123, "key3": True}
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001",
            source="test",
            metadata=metadata,
        )

        assert doc.metadata == metadata

    def test_metadata_max_entries(self) -> None:
        """Test metadata can have up to 50 entries."""
        metadata = {f"key{i}": f"value{i}" for i in range(50)}
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001",
            source="test",
            metadata=metadata,
        )

        assert len(doc.metadata) == 50

    def test_metadata_too_many_entries(self) -> None:
        """Test metadata with more than 50 entries is rejected."""
        metadata = {f"key{i}": f"value{i}" for i in range(51)}
        with pytest.raises(ValidationError) as exc_info:
            DocumentInput(
                content="Test",
                document_id="DOC-001",
                source="test",
                metadata=metadata,
            )

        assert "cannot contain more than 50 entries" in str(exc_info.value)

    def test_metadata_key_too_long(self) -> None:
        """Test metadata key exceeding 100 characters is rejected."""
        metadata = {"k" * 101: "value"}
        with pytest.raises(ValidationError) as exc_info:
            DocumentInput(
                content="Test",
                document_id="DOC-001",
                source="test",
                metadata=metadata,
            )

        assert "maximum length of 100 characters" in str(exc_info.value)

    def test_metadata_value_too_long(self) -> None:
        """Test metadata string value exceeding 1000 characters is rejected."""
        metadata = {"key": "v" * 1001}
        with pytest.raises(ValidationError) as exc_info:
            DocumentInput(
                content="Test",
                document_id="DOC-001",
                source="test",
                metadata=metadata,
            )

        assert "maximum length of 1000 characters" in str(exc_info.value)

    def test_metadata_none_allowed(self) -> None:
        """Test metadata can be None."""
        doc = DocumentInput(
            content="Test",
            document_id="DOC-001",
            source="test",
            metadata=None,
        )

        assert doc.metadata is None


class TestClassificationResultValidation:
    """Test ClassificationResult validation."""

    def test_classification_result_valid(self) -> None:
        """Test creating valid ClassificationResult."""
        result = ClassificationResult(
            document_id="DOC-001",
            category="Regulatory",
            confidence=0.9,
            subcategories=["SEC", "Financial"],
            risk_level="high",
            requires_review=False,
            reasoning="Clear regulatory filing.",
        )

        assert result.document_id == "DOC-001"
        assert result.category == "Regulatory"
        assert result.confidence == 0.9
        assert result.subcategories == ["SEC", "Financial"]
        assert result.risk_level == "high"
        assert result.requires_review is False
        assert result.reasoning == "Clear regulatory filing."
        assert isinstance(result.processed_at, datetime)

    def test_classification_result_categories(self) -> None:
        """Test all valid category values."""
        categories = ["Regulatory", "Compliance", "Risk", "Operational", "Other"]

        for category in categories:
            result = ClassificationResult(
                document_id="DOC-001",
                category=category,
                confidence=0.8,
                subcategories=[],
                risk_level="low",
                requires_review=False,
                reasoning="Test",
            )
            assert result.category == category

    def test_classification_result_invalid_category(self) -> None:
        """Test invalid category is rejected."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                document_id="DOC-001",
                category="InvalidCategory",
                confidence=0.8,
                subcategories=[],
                risk_level="low",
                requires_review=False,
                reasoning="Test",
            )

    def test_classification_result_risk_levels(self) -> None:
        """Test all valid risk level values."""
        risk_levels = ["low", "medium", "high", "critical"]

        for risk_level in risk_levels:
            result = ClassificationResult(
                document_id="DOC-001",
                category="Risk",
                confidence=0.8,
                subcategories=[],
                risk_level=risk_level,
                requires_review=False,
                reasoning="Test",
            )
            assert result.risk_level == risk_level

    def test_classification_result_invalid_risk_level(self) -> None:
        """Test invalid risk level is rejected."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                document_id="DOC-001",
                category="Risk",
                confidence=0.8,
                subcategories=[],
                risk_level="invalid",
                requires_review=False,
                reasoning="Test",
            )

    def test_classification_result_confidence_bounds(self) -> None:
        """Test confidence score must be between 0.0 and 1.0."""
        # Valid: 0.0
        result = ClassificationResult(
            document_id="DOC-001",
            category="Other",
            confidence=0.0,
            subcategories=[],
            risk_level="low",
            requires_review=True,
            reasoning="Test",
        )
        assert result.confidence == 0.0

        # Valid: 1.0
        result = ClassificationResult(
            document_id="DOC-002",
            category="Regulatory",
            confidence=1.0,
            subcategories=[],
            risk_level="low",
            requires_review=False,
            reasoning="Test",
        )
        assert result.confidence == 1.0

    def test_classification_result_confidence_below_zero(self) -> None:
        """Test confidence below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                document_id="DOC-001",
                category="Other",
                confidence=-0.1,
                subcategories=[],
                risk_level="low",
                requires_review=False,
                reasoning="Test",
            )

    def test_classification_result_confidence_above_one(self) -> None:
        """Test confidence above 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                document_id="DOC-001",
                category="Regulatory",
                confidence=1.1,
                subcategories=[],
                risk_level="low",
                requires_review=False,
                reasoning="Test",
            )

    def test_classification_result_empty_subcategories(self) -> None:
        """Test subcategories can be empty list."""
        result = ClassificationResult(
            document_id="DOC-001",
            category="Other",
            confidence=0.5,
            subcategories=[],
            risk_level="low",
            requires_review=True,
            reasoning="Unable to determine subcategories.",
        )

        assert result.subcategories == []

    def test_classification_result_processed_at_timezone(self) -> None:
        """Test processed_at uses UTC timezone."""
        result = ClassificationResult(
            document_id="DOC-001",
            category="Regulatory",
            confidence=0.9,
            subcategories=[],
            risk_level="low",
            requires_review=False,
            reasoning="Test",
        )

        # Should have timezone info
        assert result.processed_at.tzinfo is not None
        assert result.processed_at.tzinfo == timezone.utc

    def test_classification_result_serialization(self) -> None:
        """Test ClassificationResult can be serialized to dict."""
        result = ClassificationResult(
            document_id="DOC-001",
            category="Compliance",
            confidence=0.85,
            subcategories=["AML", "KYC"],
            risk_level="medium",
            requires_review=False,
            reasoning="Compliance requirements identified.",
        )

        result_dict = result.model_dump()

        assert result_dict["document_id"] == "DOC-001"
        assert result_dict["category"] == "Compliance"
        assert result_dict["confidence"] == 0.85
        assert result_dict["subcategories"] == ["AML", "KYC"]
        assert result_dict["risk_level"] == "medium"
