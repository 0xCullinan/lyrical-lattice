"""
File: src/api/routers/health.py
Purpose: GET /api/v1/health, /api/v1/stats, /metrics endpoints per Section 8.5, 8.6, REQ-API-014
"""

from datetime import datetime
import time
from fastapi import APIRouter, Response, Depends
from fastapi.responses import PlainTextResponse

from src.api.models.responses import (
    HealthResponse,
    HealthStatus,
    ServiceHealth,
    StatsResponse,
)
from src.api.dependencies import get_db, get_cache, get_faiss_engine
from src.services.db_service import DatabaseService
from src.services.cache_service import CacheService
from src.core.search.faiss_engine import FAISSEngine
from src.utils.metrics import metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Track startup time
_startup_time = time.time()


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all system components.",
    responses={
        200: {"description": "All services healthy"},
        503: {"description": "One or more services unhealthy"},
    },
)
async def health_check(
    db: DatabaseService = Depends(get_db),
    cache: CacheService = Depends(get_cache),
    faiss: FAISSEngine = Depends(get_faiss_engine),
) -> HealthResponse:
    """Check health of all services.
    
    Returns:
        HealthResponse with status of each service.
    """
    services = []
    all_healthy = True
    
    # Check PostgreSQL
    db_healthy, db_latency = await db.check_health()
    services.append(ServiceHealth(
        name="PostgreSQL",
        status=HealthStatus.HEALTHY if db_healthy else HealthStatus.UNHEALTHY,
        latency_ms=db_latency,
        error=None if db_healthy else "Connection failed",
    ))
    if not db_healthy:
        all_healthy = False
    
    # Check Redis
    cache_healthy, cache_latency = await cache.check_health()
    services.append(ServiceHealth(
        name="Redis",
        status=HealthStatus.HEALTHY if cache_healthy else HealthStatus.UNHEALTHY,
        latency_ms=cache_latency,
        error=None if cache_healthy else "Connection failed",
    ))
    if not cache_healthy:
        all_healthy = False
    
    # Check FAISS
    faiss_healthy, faiss_latency = faiss.check_health()
    services.append(ServiceHealth(
        name="FAISS Index",
        status=HealthStatus.HEALTHY if faiss_healthy else HealthStatus.UNHEALTHY,
        latency_ms=faiss_latency,
        error=None if faiss_healthy else "Index not loaded",
    ))
    if not faiss_healthy:
        all_healthy = False
    
    # Check G2P (simplified - just check if it's been initialized)
    from src.api.dependencies import _g2p_engine
    g2p_loaded = _g2p_engine is not None and _g2p_engine.is_loaded()
    services.append(ServiceHealth(
        name="G2P Engine",
        status=HealthStatus.HEALTHY if g2p_loaded else HealthStatus.UNHEALTHY,
        latency_ms=None,
        error=None if g2p_loaded else "Model not loaded",
    ))
    if not g2p_loaded:
        all_healthy = False
    
    response = HealthResponse(
        overall_status=HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY,
        services=services,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
    
    return response


@router.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Get system statistics including corpus size, request counts, and cache performance.",
)
async def get_stats(
    db: DatabaseService = Depends(get_db),
    cache: CacheService = Depends(get_cache),
    faiss: FAISSEngine = Depends(get_faiss_engine),
) -> StatsResponse:
    """Get system statistics.
    
    Returns:
        StatsResponse with system metrics.
    """
    # Get corpus size from FAISS
    corpus_size = faiss.get_size()
    
    # Get cache stats
    cache_stats = await cache.get_stats()
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    cache_hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    
    # Calculate uptime
    uptime_seconds = int(time.time() - _startup_time)
    
    return StatsResponse(
        corpus_size=corpus_size,
        total_requests=0,  # Would be tracked via metrics
        avg_response_time_ms=0.0,  # Would be calculated from metrics
        cache_hit_rate=cache_hit_rate,
        uptime_seconds=uptime_seconds,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Export Prometheus-formatted metrics.",
    response_class=PlainTextResponse,
)
async def prometheus_metrics() -> Response:
    """Export Prometheus metrics per REQ-API-014.
    
    Returns:
        Prometheus-formatted metrics text.
    """
    metrics_data = metrics.export()
    return Response(
        content=metrics_data,
        media_type="text/plain; charset=utf-8",
    )
