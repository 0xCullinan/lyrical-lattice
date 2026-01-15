# LyricalLattice - Claude Code Instructions

> This file contains project conventions and learnings. Update it whenever Claude does something incorrectly.

## Project Overview

LyricalLattice is an AI-powered wordplay discovery platform with 21 wordplay detectors. Built with FastAPI (backend) and vanilla JS (frontend).

## Architecture

```
src/
├── api/routers/        # FastAPI endpoints
├── core/g2p/           # Grapheme-to-Phoneme (ByT5)
├── core/search/        # FAISS vector search
├── detection/          # 21 wordplay detectors
│   ├── phonetic_detectors.py   # 13 phonetic devices
│   ├── semantic_detectors.py   # 4 semantic devices
│   ├── music_detectors.py      # 4 music devices
│   └── unified_detector.py     # Orchestrator
├── services/           # Business logic
└── utils/              # Helpers
```

## Code Conventions

### Python
- Use type hints for all function signatures
- Docstrings: Google style
- Max line length: 100 chars
- Use `async def` for API endpoints
- Pydantic models for request/response validation

### Frontend (Vanilla JS)
- No frameworks - keep it simple
- Use CSS custom properties for theming
- ARIA labels for accessibility
- `escapeHtml()` for all user-generated content (XSS prevention)

## Common Mistakes to Avoid

### DO NOT:
- Import torch at module level (lazy load for cold start performance)
- Use `print()` for logging - use `logging` module
- Hardcode phoneme patterns - use lexicons in `data/lexicons/`
- Forget to handle empty input in detectors
- Add comments to code you didn't write
- Create new files when editing existing ones would work

### ALWAYS:
- Run `pytest` before committing
- Use the existing G2P service, don't reinvent phonemization
- Check `data/indices/` for prebuilt FAISS indices
- Handle both ARPAbet (list) and IPA (string) phoneme formats
- Use confidence scores between 0.0 and 1.0

## Testing

```bash
# Run all tests
pytest

# Run specific detector tests
pytest tests/test_detection/ -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect_wordplay` | POST | Full 21-device detection |
| `/api/v1/phonemize` | POST | Text to phonemes |
| `/api/v1/find_oronyms` | POST | Oronym search |
| `/api/v1/health` | GET | Health check |

## Performance Notes

- G2P model (ByT5) takes ~500ms cold, ~50ms warm
- FAISS search is <10ms for 644K vectors
- Target P95 latency: <200ms for full detection

## Lexicons

- `data/lexicons/entendre_lexicon.json` - Double entendre patterns
- `data/lexicons/onomatopoeia.json` - Sound words by category

## When Adding New Detectors

1. Create match type in `src/detection/models.py`
2. Implement detector class with `detect()` method
3. Register in `UnifiedWordplayDetector`
4. Add to `DeviceType` enum
5. Write tests in `tests/test_detection/`
6. Update frontend tab in `frontend/index.html`

## Useful Commands

```bash
# Start dev server
uvicorn src.api.main:app --reload --port 8000

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/

# Build indices (if needed)
python scripts/build_indices.py
```

---

## Learnings (Compounding Engineering)

> Add entries here whenever Claude makes a mistake. This helps Claude learn and avoid repeating errors.

### Format
```
### YYYY-MM-DD
- **Issue**: What went wrong
- **Fix**: What Claude should do instead
```

### Log

*(No entries yet - add mistakes here as they occur)*

---
*Last updated: January 2026*
