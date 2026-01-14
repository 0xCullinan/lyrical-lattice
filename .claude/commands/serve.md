# /serve - Start Development Server

Start the FastAPI development server with hot reload.

## Instructions

1. Start the server:
   ```bash
   cd /Users/macbook/.gemini/antigravity/scratch/oronym-assistant && uvicorn src.api.main:app --reload --port 8000
   ```

2. Server will be available at:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Frontend: Open `frontend/index.html` in browser

3. The `--reload` flag enables hot reload for development.

## Notes

- First request may be slow (model loading)
- Check `/api/v1/health` to verify server is ready
- Frontend expects API at `http://localhost:8000/api/v1`
