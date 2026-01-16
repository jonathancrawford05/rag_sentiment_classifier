FROM python:3.11-slim

ENV POETRY_VERSION=1.8.3
WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN poetry config virtualenvs.create false \
    && poetry install --only main

EXPOSE 8081
CMD ["uvicorn", "rag_sentiment_classifier.api:app", "--host", "0.0.0.0", "--port", "8081"]
