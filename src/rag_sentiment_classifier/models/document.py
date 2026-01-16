from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DocumentInput(BaseModel):
    """
    Input document for classification.

    Attributes:
        content: The document text content (1-50,000 chars)
        document_id: Unique document identifier (alphanumeric, dash, underscore only)
        source: Source system or origin of the document
        metadata: Optional metadata dictionary with size limits
    """

    content: str = Field(..., min_length=1, max_length=50000, description="Document text content")
    document_id: str = Field(
        ..., pattern=r"^[A-Za-z0-9-_]{1,100}$", description="Unique document ID"
    )
    source: str = Field(..., min_length=1, max_length=100, description="Document source system")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Validate and sanitize document content."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be empty after stripping whitespace")

        # Remove null bytes and other potentially problematic control characters
        # Keep newlines, returns, and tabs as they're legitimate in documents
        sanitized = "".join(char for char in stripped if ord(char) >= 32 or char in "\n\r\t")

        if not sanitized:
            raise ValueError("Content contains only invalid characters")

        return sanitized

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate metadata structure and size."""
        if value is None:
            return value

        # Limit number of metadata entries
        if len(value) > 50:
            raise ValueError("Metadata cannot contain more than 50 entries")

        # Validate each key-value pair
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError("Metadata keys must be strings")
            if len(key) > 100:
                raise ValueError(f"Metadata key '{key}' exceeds maximum length of 100 characters")

            # Validate value types and sizes
            if isinstance(val, str) and len(val) > 1000:
                raise ValueError(
                    f"Metadata value for key '{key}' exceeds maximum length of 1000 characters"
                )
            elif isinstance(val, list | dict) and len(str(val)) > 1000:
                raise ValueError(f"Metadata value for key '{key}' is too large")

        return value


class ClassificationResult(BaseModel):
    """
    Structured classification output.

    Attributes:
        document_id: Document identifier
        category: Primary classification category
        confidence: Confidence score (0.0 to 1.0)
        subcategories: List of subcategory tags
        risk_level: Assessed risk level
        requires_review: Whether manual review is recommended
        reasoning: Explanation of the classification decision
        processed_at: Timestamp when classification was performed
    """

    document_id: str
    category: Literal["Regulatory", "Compliance", "Risk", "Operational", "Other"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    subcategories: list[str] = Field(
        default_factory=list, description="Classification subcategories"
    )
    risk_level: Literal["low", "medium", "high", "critical"]
    requires_review: bool
    reasoning: str
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Processing timestamp"
    )
