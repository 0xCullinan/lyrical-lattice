# Oronym & Lyric Assistant

A phonetic search and analysis system for finding oronyms, rhymes, and phonetically similar phrases.

## Features

- **Text-to-Phoneme (G2P)**: Convert text to IPA using ByT5 model with g2p-en fallback
- **Audio-to-Phoneme (S2P)**: Transcribe audio to phonemes (Conformer-CTC)
- **Oronym Detection**: Find phrases that sound identical but have different meanings
- **Rhyme Finding**: Perfect, near, assonance, consonance, and multisyllabic rhymes
- **Cross-lingual Support**: Multiple language phonemization

## Prerequisites

- Python 3.11.6 or 3.11.7
- Docker & Docker Compose
- PostgreSQL 16.2
- Redis 7.2.4

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd oronym-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Start Services with Docker

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head
```

### 3. Run the API

```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access the API

- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/api/v1/health
- **Metrics**: http://localhost:8000/metrics

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/phonemize` | POST | Convert text to IPA |
| `/api/v1/find_oronyms` | POST | Find phonetically similar phrases |
| `/api/v1/find_rhymes` | POST | Find rhyming words |
| `/api/v1/audio/transcribe` | POST | Transcribe audio to phonemes |
| `/api/v1/health` | GET | Health check |
| `/api/v1/stats` | GET | System statistics |

## Example Usage

### Phonemize Text

```bash
curl -X POST http://localhost:8000/api/v1/phonemize \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "language": "en_US"}'
```

### Find Oronyms

```bash
curl -X POST http://localhost:8000/api/v1/find_oronyms \
  -H "Content-Type: application/json" \
  -d '{"text": "ice cream", "max_results": 10}'
```

### Find Rhymes

```bash
curl -X POST http://localhost:8000/api/v1/find_rhymes \
  -H "Content-Type: application/json" \
  -d '{"word": "nation", "rhyme_type": "perfect"}'
```

## Project Structure

```
oronym-assistant/
├── src/
│   ├── api/           # FastAPI application
│   ├── core/          # Business logic (G2P, Audio, Search)
│   ├── models/        # SQLAlchemy ORM models
│   ├── services/      # External services (DB, Cache, Files)
│   └── utils/         # Utilities
├── tests/             # Unit and integration tests
├── scripts/           # Data pipeline scripts
├── data/              # Models, corpus, indices
├── alembic/           # Database migrations
└── k8s/               # Kubernetes manifests
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```

## Building for Production

```bash
# Build Docker image
docker build -t oronym-assistant:latest .

# Run with Docker Compose (all services)
docker-compose up -d
```

## Configuration

See `.env.example` for all configuration options. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql+asyncpg://... | PostgreSQL connection |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `LOG_LEVEL` | INFO | Logging level |
| `RATE_LIMIT_PER_MINUTE` | 100 | API rate limit |
| `CACHE_TTL_SECONDS` | 3600 | Cache TTL |

## License

MIT
