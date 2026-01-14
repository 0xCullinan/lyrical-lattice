"""
File: src/utils/metrics.py
Purpose: Prometheus metrics collection per REQ-API-014
"""

from prometheus_client import Counter, Histogram, Gauge, Info, REGISTRY
from prometheus_client.exposition import generate_latest


# Request metrics
REQUEST_COUNT = Counter(
    "oronym_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "oronym_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0],
)

# Processing metrics
G2P_LATENCY = Histogram(
    "oronym_g2p_latency_seconds",
    "G2P processing latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)

AUDIO_PROCESSING_LATENCY = Histogram(
    "oronym_audio_processing_latency_seconds",
    "Audio processing latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

SEARCH_LATENCY = Histogram(
    "oronym_search_latency_seconds",
    "Search latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

# Cache metrics
CACHE_HITS = Counter(
    "oronym_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "oronym_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    "oronym_active_connections",
    "Number of active database connections",
)

CORPUS_SIZE = Gauge(
    "oronym_corpus_size",
    "Number of entries in the corpus",
)

MODEL_LOADED = Gauge(
    "oronym_model_loaded",
    "Whether a model is loaded",
    ["model_name"],
)

# Application info
APP_INFO = Info(
    "oronym_app",
    "Application information",
)


class MetricsCollector:
    """Centralized metrics collection and export.
    
    Provides methods to record metrics and export Prometheus format.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        APP_INFO.info({
            "version": "1.0.0",
            "python_version": "3.11",
        })
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Record an API request.
        
        Args:
            method: HTTP method.
            endpoint: API endpoint path.
            status_code: HTTP response status code.
            latency_seconds: Request latency in seconds.
        """
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(latency_seconds)
    
    def record_g2p_latency(self, latency_seconds: float) -> None:
        """Record G2P processing latency.
        
        Args:
            latency_seconds: Processing time in seconds.
        """
        G2P_LATENCY.observe(latency_seconds)
    
    def record_audio_latency(self, latency_seconds: float) -> None:
        """Record audio processing latency.
        
        Args:
            latency_seconds: Processing time in seconds.
        """
        AUDIO_PROCESSING_LATENCY.observe(latency_seconds)
    
    def record_search_latency(self, latency_seconds: float) -> None:
        """Record search latency.
        
        Args:
            latency_seconds: Processing time in seconds.
        """
        SEARCH_LATENCY.observe(latency_seconds)
    
    def record_cache_hit(self, cache_type: str = "redis") -> None:
        """Record a cache hit.
        
        Args:
            cache_type: Type of cache (redis, memory).
        """
        CACHE_HITS.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "redis") -> None:
        """Record a cache miss.
        
        Args:
            cache_type: Type of cache (redis, memory).
        """
        CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    def set_corpus_size(self, size: int) -> None:
        """Set the corpus size metric.
        
        Args:
            size: Number of entries in corpus.
        """
        CORPUS_SIZE.set(size)
    
    def set_model_loaded(self, model_name: str, loaded: bool = True) -> None:
        """Set model loaded status.
        
        Args:
            model_name: Name of the model.
            loaded: Whether model is loaded.
        """
        MODEL_LOADED.labels(model_name=model_name).set(1 if loaded else 0)
    
    def set_active_connections(self, count: int) -> None:
        """Set active database connections count.
        
        Args:
            count: Number of active connections.
        """
        ACTIVE_CONNECTIONS.set(count)
    
    def export(self) -> bytes:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics bytes.
        """
        return generate_latest(REGISTRY)


# Global metrics instance
metrics = MetricsCollector()
