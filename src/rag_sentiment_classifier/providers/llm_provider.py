"""LLM provider interface and implementations."""

from typing import Protocol, runtime_checkable

from rag_sentiment_classifier.models.document import ClassificationResult


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for Large Language Model providers.

    This interface allows swapping between different LLM implementations
    (Ollama, OpenAI, Anthropic, etc.) without changing the service logic.
    """

    async def classify(self, document_id: str, content: str) -> ClassificationResult:
        """
        Classify a document using the LLM.

        Args:
            document_id: Unique identifier for the document
            content: Document text content to classify

        Returns:
            ClassificationResult with category, confidence, and risk assessment

        Raises:
            Exception: If classification fails
        """
        ...
