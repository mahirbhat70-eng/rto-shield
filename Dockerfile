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

# Set production environment defaults with cryptographic artifact pinning
ENV RTO_SHIELD_ENV=production
ENV RTO_SHIELD_PINNED_DIGESTS='{"models/tree_model.pkl":"e75478885e620297d17f08cc9e7211aa1d1385f33cfb73560bbf916a77fa7477","models/tree_model_calibrated.pkl":"2e6b0c5198dd56b2053263ea3d25835bcd1ee95b57c2c982de5650a297538b2c","models/tree_model_booster.pkl":"db099b05d7af60a74c5f4a788b435cc2ef1c89a1026a2de81258875df796f23a","models/logistic_baseline.pkl":"8530ca0636751d03b9c4cba5532d4a39908e084e75050630a5959ad0b23fd29d"}'
ENV PYTHONPATH=/app
ENV PORT=8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Start FastAPI serving API via uvicorn (respects dynamic PaaS $PORT)
CMD ["sh", "-c", "uvicorn src.serve.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
