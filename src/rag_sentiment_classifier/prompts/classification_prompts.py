from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from rag_sentiment_classifier.models.document import ClassificationResult

parser = PydanticOutputParser(pydantic_object=ClassificationResult)

CLASSIFICATION_PROMPT = PromptTemplate(
    template=(
        "You are an expert regulatory document classifier.\n\n"
        "DOCUMENT ID: {document_id}\n"
        "DOCUMENT CONTENT:\n{content}\n\n"
        "Return a JSON object that satisfies this schema.\n"
        "- Category: Regulatory, Compliance, Risk, Operational, Other\n"
        "- Confidence: 0.0 to 1.0\n"
        "- Subcategories: list of strings\n"
        "- Risk Level: low, medium, high, critical\n"
        "- Requires Review: true if confidence < 0.7\n"
        "- Reasoning: 2-3 sentences\n\n"
        "{format_instructions}"
    ),
    input_variables=["document_id", "content"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
