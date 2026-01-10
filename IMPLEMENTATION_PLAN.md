# Implementation Plan: RAG Sentiment Classifier
**Version:** 1.0
**Date:** 2026-01-10
**Status:** Planning Phase

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Project Foundation](#phase-1-project-foundation)
4. [Phase 2: Core RAG Implementation](#phase-2-core-rag-implementation)
5. [Phase 3: Sentiment Classification](#phase-3-sentiment-classification)
6. [Phase 4: Testing & Quality Assurance](#phase-4-testing--quality-assurance)
7. [Phase 5: API & Deployment](#phase-5-api--deployment)
8. [Phase 6: Optimization & Monitoring](#phase-6-optimization--monitoring)
9. [Acceptance Criteria](#acceptance-criteria)
10. [Appendix](#appendix)

---

## Overview

This document provides a detailed, step-by-step implementation plan for building a RAG (Retrieval-Augmented Generation) system for sentiment classification from scratch.

**Goal:** Create a production-ready sentiment classification system that uses RAG to provide context-aware sentiment analysis.

**Timeline:** 4-6 weeks for MVP
**Team Size:** 1-2 developers
**Complexity:** Medium

---

## Prerequisites

### Required Accounts/API Keys
- [ ] OpenAI API key OR Anthropic API key
- [ ] GitHub repository access (already configured)
- [ ] Python 3.10+ installed locally

### Development Environment
- [ ] Python 3.10 or higher
- [ ] Git configured
- [ ] Code editor (VSCode, PyCharm, etc.)
- [ ] Docker (optional, for containerization)

---

## Phase 1: Project Foundation

**Duration:** 2-3 days
**Priority:** 🔴 CRITICAL
**Dependencies:** None

### 1.1 Basic Project Structure

**Task:** Create directory structure and essential files

**Action Items:**
```bash
# Create directory structure
mkdir -p src/{config,models,services,pipeline,utils,api}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p data/{raw,processed,vector_store}
mkdir -p scripts
mkdir -p docs
mkdir -p notebooks

# Create __init__.py files
touch src/__init__.py
touch src/config/__init__.py
touch src/models/__init__.py
touch src/services/__init__.py
touch src/pipeline/__init__.py
touch src/utils/__init__.py
touch src/api/__init__.py
touch tests/__init__.py
```

**Files to Create:**
- [x] Directory structure (as above)
- [ ] `.gitignore`
- [ ] `.env.example`
- [ ] `pyproject.toml` or `requirements.txt`
- [ ] `setup.py` (optional)
- [ ] `LICENSE`

**Deliverables:**
- ✅ Complete folder structure
- ✅ Git ignore rules configured
- ✅ Environment template created

### 1.2 Dependency Management

**Task:** Set up Python dependencies

**Option A: Using Poetry (Recommended)**
```bash
poetry init
poetry add langchain chromadb sentence-transformers openai pydantic pydantic-settings python-dotenv loguru
poetry add --group dev pytest pytest-cov black flake8 mypy pre-commit
```

**Option B: Using pip**
```bash
# Create requirements.txt
pip install -r requirements.txt
```

**requirements.txt:**
```txt
# Core Dependencies
langchain>=0.1.0
chromadb>=0.4.22
sentence-transformers>=2.2.2
openai>=1.10.0
transformers>=4.36.0
torch>=2.1.0

# Configuration
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0

# Utilities
loguru>=0.7.2
tenacity>=8.2.3
tqdm>=4.66.0

# Optional API
fastapi>=0.109.0
uvicorn>=0.27.0

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
black>=23.12.0
flake8>=7.0.0
mypy>=1.8.0
pre-commit>=3.6.0
```

**Deliverables:**
- ✅ All dependencies installed
- ✅ Virtual environment activated
- ✅ Lock file generated

### 1.3 Configuration Management

**Task:** Create configuration system

**File:** `src/config/settings.py`
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "RAG Sentiment Classifier"
    debug: bool = False

    # LLM Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.0

    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Vector Store
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "documents"

    # RAG Settings
    max_retrieved_docs: int = 5
    similarity_threshold: float = 0.7

    # Sentiment Settings
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

**File:** `.env.example`
```env
# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Application Settings
DEBUG=false
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.0

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_PATH=./data/vector_store
COLLECTION_NAME=documents

# RAG Configuration
MAX_RETRIEVED_DOCS=5
SIMILARITY_THRESHOLD=0.7

# Sentiment Model
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
```

**Deliverables:**
- ✅ Settings class with validation
- ✅ Environment template
- ✅ Configuration documentation

### 1.4 Logging & Error Handling

**Task:** Set up logging infrastructure

**File:** `src/utils/logger.py`
```python
import sys
from loguru import logger
from src.config.settings import settings

def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if settings.debug else "INFO",
    )
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
    )
    return logger

log = setup_logger()
```

**File:** `src/utils/exceptions.py`
```python
class RAGException(Exception):
    """Base exception for RAG system"""
    pass

class EmbeddingException(RAGException):
    """Exception for embedding generation failures"""
    pass

class RetrievalException(RAGException):
    """Exception for document retrieval failures"""
    pass

class LLMException(RAGException):
    """Exception for LLM service failures"""
    pass

class SentimentException(RAGException):
    """Exception for sentiment analysis failures"""
    pass
```

**Deliverables:**
- ✅ Structured logging system
- ✅ Custom exception hierarchy
- ✅ Log rotation configured

### 1.5 Documentation

**Task:** Expand README and create documentation structure

**Update:** `README.md`
```markdown
# RAG Sentiment Classifier

A Retrieval-Augmented Generation (RAG) system for context-aware sentiment classification.

## Features
- Semantic document retrieval using vector embeddings
- LLM-powered sentiment analysis with contextual understanding
- Configurable sentiment models and thresholds
- RESTful API (optional)
- Docker support

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key or Anthropic API key

### Setup
1. Clone the repository
   git clone <repo-url>
   cd rag_sentiment_classifier

2. Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Configure environment
   cp .env.example .env
   # Edit .env and add your API keys

5. Initialize vector database
   python scripts/setup_database.py

## Usage

### Quick Start
python src/main.py --query "This product is amazing!"

### API Server
uvicorn src.api.routes:app --reload

## Documentation
- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Contributing](CONTRIBUTING.md)

## License
MIT
```

**Create:** `CONTRIBUTING.md`

**Deliverables:**
- ✅ Comprehensive README
- ✅ Contributing guidelines
- ✅ Documentation structure

---

## Phase 2: Core RAG Implementation

**Duration:** 5-7 days
**Priority:** 🔴 CRITICAL
**Dependencies:** Phase 1 completed

### 2.1 Data Models

**Task:** Define data structures

**File:** `src/models/document.py`
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RetrievedDocument(Document):
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Ranking position")
```

**File:** `src/models/sentiment.py`
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime

SentimentLabel = Literal["positive", "negative", "neutral"]

class SentimentResult(BaseModel):
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    text: str
    retrieved_docs: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SentimentAnalysis(BaseModel):
    query: str
    result: SentimentResult
    processing_time: float
    metadata: dict = Field(default_factory=dict)
```

**Deliverables:**
- ✅ Document models with validation
- ✅ Sentiment models with type safety
- ✅ Pydantic integration

### 2.2 Embedding Service

**Task:** Implement text embedding generation

**File:** `src/services/embeddings.py`
```python
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from src.config.settings import settings
from src.utils.logger import log
from src.utils.exceptions import EmbeddingException

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model
        try:
            self.model = SentenceTransformer(self.model_name)
            log.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            raise EmbeddingException(f"Failed to load embedding model: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            log.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            raise EmbeddingException(f"Embedding generation failed: {e}")

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
```

**Deliverables:**
- ✅ Embedding service class
- ✅ Error handling
- ✅ Logging integration

### 2.3 Vector Store Integration

**Task:** Set up ChromaDB for document storage

**File:** `src/services/vector_store.py`
```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.models.document import Document, RetrievedDocument
from src.utils.logger import log
from src.utils.exceptions import RetrievalException

class VectorStore:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(
            path=settings.vector_store_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        log.info(f"Initialized vector store at {settings.vector_store_path}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        try:
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_service.encode(texts)

            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=[doc.id for doc in documents],
                metadatas=[doc.metadata for doc in documents]
            )
            log.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            raise RetrievalException(f"Failed to add documents: {e}")

    def search(self, query: str, top_k: int = None) -> List[RetrievedDocument]:
        """Search for similar documents"""
        try:
            top_k = top_k or settings.max_retrieved_docs
            query_embedding = self.embedding_service.encode(query)

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            retrieved_docs = []
            for i, (doc_id, content, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                score = 1 - distance  # Convert distance to similarity
                if score >= settings.similarity_threshold:
                    retrieved_docs.append(RetrievedDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        score=score,
                        rank=i+1
                    ))

            log.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return retrieved_docs
        except Exception as e:
            raise RetrievalException(f"Search failed: {e}")

    def count(self) -> int:
        """Get document count"""
        return self.collection.count()
```

**Deliverables:**
- ✅ Vector store service
- ✅ Document insertion
- ✅ Similarity search

### 2.4 LLM Service

**Task:** Integrate LLM for text generation

**File:** `src/services/llm_service.py`
```python
from openai import OpenAI
from typing import List, Optional
from src.config.settings import settings
from src.models.document import RetrievedDocument
from src.utils.logger import log
from src.utils.exceptions import LLMException

class LLMService:
    def __init__(self):
        if not settings.openai_api_key:
            raise LLMException("OpenAI API key not configured")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        log.info(f"Initialized LLM service with model: {self.model}")

    def generate_with_context(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with retrieved context"""
        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Document {doc.rank} (score: {doc.score:.2f}):\n{doc.content}"
                for doc in retrieved_docs
            ])

            # Default system prompt
            if not system_prompt:
                system_prompt = (
                    "You are a sentiment analysis assistant. "
                    "Analyze the sentiment of the user's query using the provided context documents. "
                    "Respond with only: POSITIVE, NEGATIVE, or NEUTRAL, followed by a confidence score (0-1)."
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=100
            )

            result = response.choices[0].message.content
            log.debug(f"LLM response: {result}")
            return result

        except Exception as e:
            raise LLMException(f"LLM generation failed: {e}")
```

**Deliverables:**
- ✅ LLM client integration
- ✅ Context-aware prompting
- ✅ Response generation

---

## Phase 3: Sentiment Classification

**Duration:** 3-4 days
**Priority:** ⚠️ HIGH
**Dependencies:** Phase 2 completed

### 3.1 Sentiment Classifier

**Task:** Implement sentiment analysis logic

**File:** `src/services/sentiment_classifier.py`
```python
from transformers import pipeline
from typing import Tuple
from src.config.settings import settings
from src.utils.logger import log
from src.utils.exceptions import SentimentException

class SentimentClassifier:
    def __init__(self):
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model,
                device=-1  # CPU, use 0 for GPU
            )
            log.info(f"Loaded sentiment model: {settings.sentiment_model}")
        except Exception as e:
            raise SentimentException(f"Failed to load sentiment model: {e}")

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify sentiment of text"""
        try:
            result = self.classifier(text)[0]
            label = result['label'].lower()
            confidence = result['score']

            # Normalize label
            if 'pos' in label:
                label = 'positive'
            elif 'neg' in label:
                label = 'negative'
            else:
                label = 'neutral'

            log.debug(f"Sentiment: {label} ({confidence:.2f})")
            return label, confidence

        except Exception as e:
            raise SentimentException(f"Classification failed: {e}")
```

**Deliverables:**
- ✅ Sentiment classifier service
- ✅ Pre-trained model integration
- ✅ Label normalization

### 3.2 RAG Pipeline

**Task:** Orchestrate all components

**File:** `src/pipeline/rag_pipeline.py`
```python
import time
from typing import Optional
from src.services.embeddings import EmbeddingService
from src.services.vector_store import VectorStore
from src.services.llm_service import LLMService
from src.services.sentiment_classifier import SentimentClassifier
from src.models.sentiment import SentimentResult, SentimentAnalysis
from src.utils.logger import log

class RAGPipeline:
    def __init__(self):
        log.info("Initializing RAG pipeline...")
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(self.embedding_service)
        self.llm_service = LLMService()
        self.sentiment_classifier = SentimentClassifier()
        log.info("RAG pipeline initialized successfully")

    def analyze_sentiment(
        self,
        query: str,
        use_rag: bool = True
    ) -> SentimentAnalysis:
        """Analyze sentiment with optional RAG context"""
        start_time = time.time()

        try:
            retrieved_docs = []

            if use_rag:
                # Retrieve relevant documents
                retrieved_docs = self.vector_store.search(query)
                log.info(f"Retrieved {len(retrieved_docs)} documents")

                if retrieved_docs:
                    # Use LLM with context
                    llm_response = self.llm_service.generate_with_context(
                        query, retrieved_docs
                    )
                    # Parse LLM response (simplified)
                    label, confidence = self._parse_llm_response(llm_response)
                else:
                    log.warning("No relevant documents found, using direct classification")
                    label, confidence = self.sentiment_classifier.classify(query)
            else:
                # Direct sentiment classification
                label, confidence = self.sentiment_classifier.classify(query)

            processing_time = time.time() - start_time

            result = SentimentResult(
                label=label,
                confidence=confidence,
                text=query,
                retrieved_docs=[doc.id for doc in retrieved_docs] if retrieved_docs else None
            )

            analysis = SentimentAnalysis(
                query=query,
                result=result,
                processing_time=processing_time,
                metadata={
                    "num_docs_retrieved": len(retrieved_docs),
                    "used_rag": use_rag
                }
            )

            log.info(f"Analysis complete in {processing_time:.2f}s: {label} ({confidence:.2f})")
            return analysis

        except Exception as e:
            log.error(f"Pipeline error: {e}")
            raise

    def _parse_llm_response(self, response: str) -> tuple[str, float]:
        """Parse LLM response to extract sentiment and confidence"""
        # Simplified parser - enhance as needed
        response = response.lower()

        if 'positive' in response:
            label = 'positive'
        elif 'negative' in response:
            label = 'negative'
        else:
            label = 'neutral'

        # Extract confidence (basic implementation)
        try:
            confidence = float([s for s in response.split() if s.replace('.','').isdigit()][0])
        except:
            confidence = 0.8  # Default

        return label, confidence
```

**Deliverables:**
- ✅ Integrated RAG pipeline
- ✅ Component orchestration
- ✅ Performance tracking

### 3.3 Main Entry Point

**Task:** Create application entry point

**File:** `src/main.py`
```python
import argparse
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import log

def main():
    parser = argparse.ArgumentParser(description="RAG Sentiment Classifier")
    parser.add_argument("--query", type=str, required=True, help="Text to analyze")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline()
        result = pipeline.analyze_sentiment(args.query, use_rag=not args.no_rag)

        print(f"\nQuery: {result.query}")
        print(f"Sentiment: {result.result.label.upper()}")
        print(f"Confidence: {result.result.confidence:.2%}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.result.retrieved_docs:
            print(f"Documents Used: {len(result.result.retrieved_docs)}")

    except Exception as e:
        log.error(f"Application error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
```

**Deliverables:**
- ✅ CLI interface
- ✅ Argument parsing
- ✅ Error handling

---

## Phase 4: Testing & Quality Assurance

**Duration:** 4-5 days
**Priority:** ⚠️ HIGH
**Dependencies:** Phase 3 completed

### 4.1 Test Infrastructure

**Task:** Set up pytest and fixtures

**File:** `tests/conftest.py`
```python
import pytest
from src.services.embeddings import EmbeddingService
from src.models.document import Document

@pytest.fixture
def sample_documents():
    return [
        Document(id="1", content="This product is excellent!", metadata={"source": "review"}),
        Document(id="2", content="Terrible experience, very disappointed.", metadata={"source": "review"}),
        Document(id="3", content="It's okay, nothing special.", metadata={"source": "review"}),
    ]

@pytest.fixture
def embedding_service():
    return EmbeddingService()
```

**File:** `pytest.ini`
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=src --cov-report=html --cov-report=term
```

**Deliverables:**
- ✅ Test configuration
- ✅ Shared fixtures
- ✅ Coverage setup

### 4.2 Unit Tests

**Task:** Write comprehensive unit tests

**File:** `tests/unit/test_embeddings.py`
**File:** `tests/unit/test_sentiment.py`
**File:** `tests/unit/test_vector_store.py`

**Deliverables:**
- ✅ 80%+ code coverage
- ✅ All services tested
- ✅ Edge cases covered

### 4.3 Integration Tests

**Task:** Test end-to-end workflows

**File:** `tests/integration/test_rag_pipeline.py`
```python
import pytest
from src.pipeline.rag_pipeline import RAGPipeline
from src.models.document import Document

def test_rag_pipeline_sentiment_analysis():
    pipeline = RAGPipeline()

    # Add test documents
    docs = [
        Document(id="test1", content="Amazing product, highly recommend!"),
        Document(id="test2", content="Worst purchase ever made."),
    ]
    pipeline.vector_store.add_documents(docs)

    # Test positive sentiment
    result = pipeline.analyze_sentiment("I love this product!")
    assert result.result.label == "positive"
    assert result.result.confidence > 0.5

    # Test negative sentiment
    result = pipeline.analyze_sentiment("This is terrible")
    assert result.result.label == "negative"
```

**Deliverables:**
- ✅ Pipeline integration tests
- ✅ End-to-end workflows tested
- ✅ Performance benchmarks

### 4.4 Code Quality

**Task:** Configure linting and formatting

**File:** `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**File:** `pyproject.toml` (Black config)
```toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Deliverables:**
- ✅ Pre-commit hooks configured
- ✅ Code formatted with Black
- ✅ Type hints validated with mypy

---

## Phase 5: API & Deployment

**Duration:** 3-4 days
**Priority:** 📋 MEDIUM
**Dependencies:** Phase 4 completed

### 5.1 REST API (Optional)

**Task:** Create FastAPI endpoints

**File:** `src/api/schemas.py`
```python
from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str
    use_rag: bool = True

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    processing_time: float
```

**File:** `src/api/routes.py`
```python
from fastapi import FastAPI, HTTPException
from src.api.schemas import SentimentRequest, SentimentResponse
from src.pipeline.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Sentiment Classifier")
pipeline = RAGPipeline()

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = pipeline.analyze_sentiment(request.text, request.use_rag)
        return SentimentResponse(
            sentiment=result.result.label,
            confidence=result.result.confidence,
            processing_time=result.processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Deliverables:**
- ✅ REST API endpoints
- ✅ Request/response validation
- ✅ API documentation (auto-generated)

### 5.2 Containerization

**Task:** Create Docker setup

**File:** `Dockerfile`
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]
```

**File:** `docker-compose.yml`
```yaml
version: '3.8'

services:
  rag-sentiment:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
```

**Deliverables:**
- ✅ Dockerfile created
- ✅ Docker Compose setup
- ✅ Volume mapping configured

### 5.3 CI/CD Pipeline

**Task:** Set up GitHub Actions

**File:** `.github/workflows/test.yml`
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
      - name: Check code quality
        run: |
          black --check .
          flake8 .
```

**Deliverables:**
- ✅ Automated testing
- ✅ Code quality checks
- ✅ CI pipeline configured

---

## Phase 6: Optimization & Monitoring

**Duration:** Ongoing
**Priority:** 📋 LOW
**Dependencies:** Phase 5 completed

### 6.1 Performance Optimization

**Tasks:**
- [ ] Implement embedding caching
- [ ] Batch processing for multiple queries
- [ ] Optimize vector search parameters
- [ ] Profile and optimize bottlenecks

### 6.2 Monitoring

**Tasks:**
- [ ] Add metrics collection (Prometheus)
- [ ] Implement health checks
- [ ] Set up error tracking (Sentry)
- [ ] Create dashboard (Grafana)

### 6.3 Documentation

**Tasks:**
- [ ] Write architecture documentation
- [ ] Create API reference
- [ ] Add usage examples
- [ ] Record demo video

---

## Acceptance Criteria

### Functional Requirements
- ✅ System can ingest and store documents
- ✅ Semantic search returns relevant documents
- ✅ Sentiment analysis produces accurate classifications
- ✅ RAG pipeline integrates all components
- ✅ CLI interface works correctly
- ✅ API endpoints respond correctly (if implemented)

### Non-Functional Requirements
- ✅ Code coverage ≥ 80%
- ✅ All tests passing
- ✅ Code formatted and linted
- ✅ Type hints on all functions
- ✅ Documentation complete
- ✅ Response time < 2s (p95)

### Quality Gates
- ✅ No critical security vulnerabilities
- ✅ All dependencies up to date
- ✅ Pre-commit hooks passing
- ✅ CI/CD pipeline green

---

## Appendix

### A. Sample Usage Scenarios

**Scenario 1: Basic Sentiment Analysis**
```bash
python src/main.py --query "This is the best product ever!"
```

**Scenario 2: Without RAG**
```bash
python src/main.py --query "I hate this" --no-rag
```

**Scenario 3: API Call**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing experience!", "use_rag": true}'
```

### B. Troubleshooting

**Issue:** Out of memory errors
**Solution:** Reduce batch size, use CPU for embeddings

**Issue:** Slow retrieval
**Solution:** Tune HNSW parameters, reduce document count

**Issue:** Low accuracy
**Solution:** Add more relevant documents, tune similarity threshold

### C. Future Enhancements

- Multi-language support
- Real-time sentiment streaming
- Custom model fine-tuning
- A/B testing framework
- Analytics dashboard
- Feedback loop for model improvement

---

**Plan Status:** Ready for Implementation
**Next Step:** Begin Phase 1 - Project Foundation
**Questions?** Refer to CODEBASE_AUDIT_REPORT.md for detailed analysis
