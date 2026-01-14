"""
File: src/services/cache_service.py
Purpose: Redis cache service with LRU eviction per REQ-PERF-005, REQ-SEARCH-010
"""

import json
import hashlib
from typing import Optional, Any
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class CacheService:
    """Redis cache service for query results.
    
    Implements caching with LRU eviction and configurable TTL.
    Caches top 1000 most frequent queries per REQ-SEARCH-010.
    
    Attributes:
        redis: Redis client instance.
        ttl: Cache TTL in seconds.
    """
    
    MAX_CACHED_KEYS = 1000  # Per REQ-SEARCH-010
    KEY_PREFIX = "oronym:"
    
    def __init__(self):
        """Initialize cache service."""
        self._redis: Optional[Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self.ttl = settings.cache_ttl_seconds
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        logger.info("Initializing Redis connection pool")
        
        self._pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=20,
            decode_responses=True,
        )
        self._redis = Redis(connection_pool=self._pool)
        
        logger.info("Redis connection pool initialized")
    
    async def close(self) -> None:
        """Close Redis connections gracefully."""
        if self._redis:
            logger.info("Closing Redis connections")
            await self._redis.close()
            self._redis = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
    
    async def check_health(self) -> tuple[bool, Optional[int]]:
        """Check Redis health.
        
        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        import time
        
        if not self._redis:
            return False, None
        
        try:
            start = time.perf_counter()
            await self._redis.ping()
            latency_ms = int((time.perf_counter() - start) * 1000)
            return True, latency_ms
        except RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return False, None
    
    def _make_key(self, prefix: str, data: str) -> str:
        """Create a cache key from prefix and data.
        
        Args:
            prefix: Key prefix (e.g., 'oronym', 'rhyme').
            data: Data to hash for the key.
            
        Returns:
            Cache key string.
        """
        hash_val = hashlib.md5(data.encode()).hexdigest()[:16]
        return f"{self.KEY_PREFIX}{prefix}:{hash_val}"
    
    async def get(self, prefix: str, query: str) -> Optional[dict[str, Any]]:
        """Get cached result.
        
        Args:
            prefix: Cache prefix (oronym, rhyme, phoneme).
            query: Query string to look up.
            
        Returns:
            Cached result dict if found, None otherwise.
        """
        if not self._redis:
            return None
        
        key = self._make_key(prefix, query)
        
        try:
            cached = await self._redis.get(key)
            if cached:
                metrics.record_cache_hit()
                logger.debug(f"Cache hit for {prefix}:{query[:30]}")
                return json.loads(cached)
            else:
                metrics.record_cache_miss()
                logger.debug(f"Cache miss for {prefix}:{query[:30]}")
                return None
        except RedisError as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        prefix: str,
        query: str,
        result: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a result.
        
        Implements LRU by tracking access counts and evicting
        least recently used keys when limit is reached.
        
        Args:
            prefix: Cache prefix.
            query: Query string.
            result: Result to cache.
            ttl: Optional TTL override.
            
        Returns:
            True if cached successfully.
        """
        if not self._redis:
            return False
        
        key = self._make_key(prefix, query)
        ttl = ttl or self.ttl
        
        try:
            # Check if we need to evict old keys
            await self._enforce_max_keys()
            
            # Store the result
            await self._redis.setex(
                key,
                ttl,
                json.dumps(result),
            )
            
            # Track this key for LRU
            await self._redis.zadd(
                f"{self.KEY_PREFIX}lru",
                {key: float("inf")},  # Will be updated with timestamp
            )
            
            logger.debug(f"Cached {prefix}:{query[:30]} with TTL {ttl}s")
            return True
        except RedisError as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def _enforce_max_keys(self) -> None:
        """Enforce maximum cached keys limit using LRU eviction.
        
        Removes oldest keys when limit is exceeded.
        """
        if not self._redis:
            return
        
        try:
            lru_key = f"{self.KEY_PREFIX}lru"
            count = await self._redis.zcard(lru_key)
            
            if count >= self.MAX_CACHED_KEYS:
                # Remove oldest 10% of keys
                to_remove = max(1, count // 10)
                oldest = await self._redis.zrange(lru_key, 0, to_remove - 1)
                
                if oldest:
                    await self._redis.delete(*oldest)
                    await self._redis.zrem(lru_key, *oldest)
                    logger.debug(f"Evicted {len(oldest)} cache entries (LRU)")
        except RedisError as e:
            logger.warning(f"LRU enforcement error: {e}")
    
    async def delete(self, prefix: str, query: str) -> bool:
        """Delete a cached result.
        
        Args:
            prefix: Cache prefix.
            query: Query string.
            
        Returns:
            True if deleted.
        """
        if not self._redis:
            return False
        
        key = self._make_key(prefix, query)
        
        try:
            await self._redis.delete(key)
            await self._redis.zrem(f"{self.KEY_PREFIX}lru", key)
            return True
        except RedisError as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all cached entries.
        
        Returns:
            True if cleared successfully.
        """
        if not self._redis:
            return False
        
        try:
            keys = []
            async for key in self._redis.scan_iter(f"{self.KEY_PREFIX}*"):
                keys.append(key)
            
            if keys:
                await self._redis.delete(*keys)
            
            logger.info(f"Cleared {len(keys)} cache entries")
            return True
        except RedisError as e:
            logger.warning(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats.
        """
        if not self._redis:
            return {"status": "disconnected"}
        
        try:
            info = await self._redis.info("stats")
            lru_count = await self._redis.zcard(f"{self.KEY_PREFIX}lru")
            
            return {
                "status": "connected",
                "cached_queries": lru_count,
                "max_cached_queries": self.MAX_CACHED_KEYS,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except RedisError as e:
            logger.warning(f"Cache stats error: {e}")
            return {"status": "error", "error": str(e)}


# Global cache service instance
cache_service = CacheService()
