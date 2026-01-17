"""Ollama LLM provider implementation."""

import logging

from langchain_community.chat_models import ChatOllama

from rag_sentiment_classifier.models.document import ClassificationResult
from rag_sentiment_classifier.prompts.classification_prompts import (
    CLASSIFICATION_PROMPT,
    parser,
)

logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    """
    Ollama-based LLM provider implementation.

    Provides classification using a local Ollama LLM instance.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout: int = 60,
    ) -> None:
        """
        Initialize Ollama LLM provider.

        Args:
            model: Ollama model name (e.g., 'llama2', 'mistral')
            base_url: Ollama server URL
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=timeout,
        )
        self.classification_chain = CLASSIFICATION_PROMPT | self.llm | parser
        logger.info(
            "OllamaLLMProvider initialized with model=%s, base_url=%s", model, base_url
        )

    async def classify(self, document_id: str, content: str) -> ClassificationResult:
        """
        Classify a document using the Ollama LLM.

        Args:
            document_id: Unique identifier for the document
            content: Document text content to classify

        Returns:
            ClassificationResult with category, confidence, and risk assessment

        Raises:
            Exception: If classification fails
        """
        logger.debug("Classifying document %s with Ollama", document_id)

        # LangChain's invoke is synchronous, but we can make it work in async context
        # by using asyncio.to_thread or ainvoke if available
        result: ClassificationResult = await self.classification_chain.ainvoke(
            {"document_id": document_id, "content": content}
        )
        result.document_id = document_id

        logger.debug(
            "Document %s classified as %s with confidence %.2f",
            document_id,
            result.category,
            result.confidence,
        )
        return result
