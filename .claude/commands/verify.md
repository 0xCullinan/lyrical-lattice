# /verify - Verify Application Works End-to-End

Test the full application stack to ensure everything works.

## Instructions

### 1. Check Server Health
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```
Expected: `{"status": "healthy"}`

### 2. Test Phonemization
```bash
curl -s -X POST http://localhost:8000/api/v1/phonemize \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}' | jq .
```
Expected: Returns phonemes array

### 3. Test Full Wordplay Detection
```bash
curl -s -X POST http://localhost:8000/api/v1/detect_wordplay \
  -H "Content-Type: application/json" \
  -d '{"text": "I scream for ice cream", "categories": ["all"]}' | jq .
```
Expected: Returns matches with oronym detection

### 4. Test Oronym Search
```bash
curl -s -X POST http://localhost:8000/api/v1/find_oronyms \
  -H "Content-Type: application/json" \
  -d '{"text": "ice cream", "max_results": 5}' | jq .
```
Expected: Returns oronym suggestions

### 5. Run Unit Tests
```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass

## Verification Checklist

- [ ] Server starts without errors
- [ ] Health endpoint returns healthy
- [ ] Phonemization works
- [ ] Wordplay detection returns results
- [ ] Oronym search finds alternatives
- [ ] All unit tests pass

## If Something Fails

1. Check server logs for errors
2. Verify FAISS indices exist in `data/indices/`
3. Check if venv is activated
4. Ensure all dependencies installed: `pip install -r requirements.txt`
