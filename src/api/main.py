"""
File: src/api/main.py
Purpose: FastAPI application initialization per REQ-API-001 to REQ-API-014
"""

import signal
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.core.config import settings
from src.api.middleware import (
    RequestIDMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
)
from src.api.routers import (
    phonemize_router,
    oronyms_router,
    rhymes_router,
    audio_router,
    health_router,
)
from src.api.dependencies import initialize_services, shutdown_services
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.
    
    Initializes services on startup and shuts down gracefully
    per REQ-API-013.
    """
    # Startup
    logger.info("Starting Oronym Assistant API")
    
    try:
        await initialize_services()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Oronym Assistant API")
    await shutdown_services()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Oronym & Lyric Assistant API",
    description=(
        "A phonetic search and analysis system for finding oronyms, rhymes, "
        "and phonetically similar phrases. Supports text and audio input with "
        "IPA transcription."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# Add CORS middleware per REQ-API-011
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
app.add_middleware(RequestIDMiddleware)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors per REQ-API-008."""
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions per REQ-API-009."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


# Register routers per REQ-API-003
app.include_router(phonemize_router)
app.include_router(oronyms_router)
app.include_router(rhymes_router)
app.include_router(audio_router)
app.include_router(health_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": "Oronym & Lyric Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=not settings.is_production,
        workers=1 if not settings.is_production else 4,
    )
