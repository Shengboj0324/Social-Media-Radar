"""Industrial-grade caching layer for performance optimization."""

import hashlib
import json
import logging
from typing import Any, Optional, Callable
from functools import wraps
import asyncio

import redis.asyncio as redis
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    key_prefix: str = "smr"  # Social Media Radar
    max_retries: int = 3


class CacheManager:
    """Industrial-grade cache manager with Redis backend.

    Features:
    - Async Redis operations
    - Automatic serialization/deserialization
    - TTL management
    - Key namespacing
    - Error handling with fallback
    - Cache invalidation
    - Batch operations
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._redis: Optional[redis.Redis] = None
        self._initialized = False

    async def _ensure_connection(self):
        """Ensure Redis connection is established."""
        if not self._initialized:
            try:
                self._redis = await redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50,
                )
                # Test connection
                await self._redis.ping()
                self._initialized = True
                logger.info("Redis cache connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._redis = None
                self._initialized = False

    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key.

        Args:
            namespace: Cache namespace (e.g., 'embeddings', 'summaries')
            key: Cache key

        Returns:
            Namespaced key
        """
        return f"{self.config.key_prefix}:{namespace}:{key}"

    def _hash_key(self, data: Any) -> str:
        """Create hash from data for cache key.

        Args:
            data: Data to hash

        Returns:
            SHA256 hash
        """
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        namespace: str,
        key: str,
        deserializer: Optional[Callable] = None,
    ) -> Optional[Any]:
        """Get value from cache.

        Args:
            namespace: Cache namespace
            key: Cache key
            deserializer: Optional function to deserialize value

        Returns:
            Cached value or None if not found
        """
        if not self.config.enabled:
            return None

        await self._ensure_connection()

        if not self._redis:
            return None

        try:
            cache_key = self._make_key(namespace, key)
            value = await self._redis.get(cache_key)

            if value is None:
                return None

            # Deserialize if needed
            if deserializer:
                return deserializer(value)

            # Try JSON deserialization
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: Optional[Callable] = None,
    ) -> bool:
        """Set value in cache.

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses config default if None)
            serializer: Optional function to serialize value

        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False

        await self._ensure_connection()

        if not self._redis:
            return False

        try:
            cache_key = self._make_key(namespace, key)
            ttl = ttl or self.config.ttl_seconds

            # Serialize value
            if serializer:
                serialized = serializer(value)
            elif isinstance(value, (dict, list)):
                serialized = json.dumps(value)
            else:
                serialized = str(value)

            # Set with TTL
            await self._redis.setex(cache_key, ttl, serialized)
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache.

        Args:
            namespace: Cache namespace
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False

        await self._ensure_connection()

        if not self._redis:
            return False

        try:
            cache_key = self._make_key(namespace, key)
            await self._redis.delete(cache_key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace.

        Args:
            namespace: Cache namespace to invalidate

        Returns:
            Number of keys deleted
        """
        if not self.config.enabled:
            return 0

        await self._ensure_connection()

        if not self._redis:
            return 0

        try:
            pattern = self._make_key(namespace, "*")
            keys = []

            # Scan for keys matching pattern
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Invalidated {deleted} keys in namespace {namespace}")
                return deleted

            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    async def get_many(
        self,
        namespace: str,
        keys: list[str],
    ) -> dict[str, Any]:
        """Get multiple values from cache.

        Args:
            namespace: Cache namespace
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        if not self.config.enabled or not keys:
            return {}

        await self._ensure_connection()

        if not self._redis:
            return {}

        try:
            cache_keys = [self._make_key(namespace, k) for k in keys]
            values = await self._redis.mget(cache_keys)

            result = {}
            for i, key in enumerate(keys):
                if values[i] is not None:
                    try:
                        result[key] = json.loads(values[i])
                    except json.JSONDecodeError:
                        result[key] = values[i]

            return result
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}

    async def set_many(
        self,
        namespace: str,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Set multiple values in cache.

        Args:
            namespace: Cache namespace
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            Number of items successfully cached
        """
        if not self.config.enabled or not items:
            return 0

        await self._ensure_connection()

        if not self._redis:
            return 0

        try:
            ttl = ttl or self.config.ttl_seconds
            count = 0

            # Use pipeline for efficiency
            async with self._redis.pipeline() as pipe:
                for key, value in items.items():
                    cache_key = self._make_key(namespace, key)
                    serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                    pipe.setex(cache_key, ttl, serialized)
                    count += 1

                await pipe.execute()

            return count
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._initialized = False


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(
    namespace: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator for caching function results.

    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from args

    Example:
        @cached(namespace="embeddings", ttl=3600)
        async def generate_embedding(text: str):
            # Expensive operation
            return embedding
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Use function name and args as key
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = cache._hash_key(":".join(key_parts))

            # Try to get from cache
            cached_value = await cache.get(namespace, cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            await cache.set(namespace, cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance.

    Returns:
        Global CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

