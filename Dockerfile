# FlipPrice AI — Production Decision Pricing Engine Container
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY src/ ./src/
COPY docs/ ./docs/
COPY reports/ ./reports/

# Set production environment defaults
ENV RTO_SHIELD_ENV=production
ENV PYTHONPATH=/app
ENV PORT=8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Start FastAPI serving API via uvicorn (respects dynamic PaaS $PORT)
CMD ["sh", "-c", "uvicorn src.serve.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
