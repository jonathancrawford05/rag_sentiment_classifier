from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentInput(BaseModel):
    """Input document for classification."""

    content: str = Field(..., min_length=1, max_length=50000)
    document_id: str
    source: str
    metadata: Optional[dict] = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Content cannot be empty")
        return stripped


class ClassificationResult(BaseModel):
    """Structured classification output."""

    document_id: str
    category: Literal["Regulatory", "Compliance", "Risk", "Operational", "Other"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    subcategories: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"]
    requires_review: bool
    reasoning: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return value


class ProcessingError(BaseModel):
    """Structured error information."""

    document_id: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
