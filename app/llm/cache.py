"""LLM-specific caching layer for response caching and deduplication.

This module provides industrial-grade caching for LLM operations:
- Response caching to reduce API calls
- Request deduplication to prevent duplicate requests
- Cost savings through cache hits
- Configurable TTL per operation type
- Cache invalidation and management
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from app.core.cache import CacheManager, get_cache_manager
from app.llm.models import EmbeddingResponse, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class LLMCacheManager:
    """LLM-specific cache manager with intelligent caching strategies.
    
    Features:
    - Response caching for identical requests
    - Embedding caching (embeddings are deterministic)
    - Request deduplication
    - Cost tracking for cache hits
    - Configurable TTL per operation type
    """
    
    # Cache namespaces
    NAMESPACE_LLM = "llm:responses"
    NAMESPACE_EMBEDDING = "llm:embeddings"
    NAMESPACE_DEDUP = "llm:dedup"
    
    # Default TTLs (in seconds)
    TTL_LLM_RESPONSE = 3600  # 1 hour for LLM responses
    TTL_EMBEDDING = 86400 * 7  # 7 days for embeddings (deterministic)
    TTL_DEDUP = 60  # 1 minute for deduplication
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize LLM cache manager.
        
        Args:
            cache_manager: Underlying cache manager (uses global if None)
        """
        self.cache = cache_manager or get_cache_manager()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cost_saved = 0.0
        
        logger.info("Initialized LLM cache manager")
    
    def _make_llm_cache_key(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """Create cache key for LLM request.
        
        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Cache key hash
        """
        # Create deterministic representation
        key_data = {
            "messages": [{"role": m.role.value if hasattr(m.role, "value") else m.role, "content": m.content} for m in messages],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        
        # Hash the data
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _make_embedding_cache_key(self, text: str, model: str) -> str:
        """Create cache key for embedding request.
        
        Args:
            text: Text to embed
            model: Model name
            
        Returns:
            Cache key hash
        """
        key_data = {"text": text, "model": model}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_llm_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Optional[LLMResponse]:
        """Get cached LLM response.
        
        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Cached response or None
        """
        cache_key = self._make_llm_cache_key(messages, model, temperature, max_tokens, **kwargs)
        
        cached_data = await self.cache.get(self.NAMESPACE_LLM, cache_key)
        
        if cached_data:
            self._cache_hits += 1
            logger.debug(f"LLM cache hit for model {model}")
            
            # Deserialize to LLMResponse
            try:
                response = LLMResponse(**cached_data)
                response.cached = True  # Mark as cached
                
                # Track cost saved
                self._cost_saved += response.cost
                
                return response
            except Exception as e:
                logger.error(f"Failed to deserialize cached LLM response: {e}")
                return None
        
        self._cache_misses += 1
        return None
    
    async def set_llm_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        response: LLMResponse,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Cache LLM response.
        
        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            response: LLM response to cache
            ttl: Time to live (uses default if None)
            **kwargs: Additional parameters
            
        Returns:
            True if cached successfully
        """
        cache_key = self._make_llm_cache_key(messages, model, temperature, max_tokens, **kwargs)
        
        # Serialize response
        response_data = response.model_dump()
        
        return await self.cache.set(
            self.NAMESPACE_LLM,
            cache_key,
            response_data,
            ttl=ttl or self.TTL_LLM_RESPONSE,
        )

    async def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[EmbeddingResponse]:
        """Get cached embedding.

        Args:
            text: Text to embed
            model: Model name

        Returns:
            Cached embedding or None
        """
        cache_key = self._make_embedding_cache_key(text, model)

        cached_data = await self.cache.get(self.NAMESPACE_EMBEDDING, cache_key)

        if cached_data:
            self._cache_hits += 1
            logger.debug(f"Embedding cache hit for model {model}")

            try:
                embedding = EmbeddingResponse(**cached_data)
                embedding.cached = True

                # Track cost saved
                self._cost_saved += embedding.usage.total_cost

                return embedding
            except Exception as e:
                logger.error(f"Failed to deserialize cached embedding: {e}")
                return None

        self._cache_misses += 1
        return None

    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: EmbeddingResponse,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache embedding.

        Args:
            text: Text that was embedded
            model: Model name
            embedding: Embedding response to cache
            ttl: Time to live (uses default if None)

        Returns:
            True if cached successfully
        """
        cache_key = self._make_embedding_cache_key(text, model)

        # Serialize embedding
        embedding_data = embedding.model_dump()

        return await self.cache.set(
            self.NAMESPACE_EMBEDDING,
            cache_key,
            embedding_data,
            ttl=ttl or self.TTL_EMBEDDING,
        )

    async def check_duplicate_request(
        self,
        request_id: str,
    ) -> bool:
        """Check if request is a duplicate.

        Args:
            request_id: Unique request identifier

        Returns:
            True if duplicate, False otherwise
        """
        cached = await self.cache.get(self.NAMESPACE_DEDUP, request_id)
        return cached is not None

    async def mark_request_processed(
        self,
        request_id: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Mark request as processed for deduplication.

        Args:
            request_id: Unique request identifier
            ttl: Time to live (uses default if None)

        Returns:
            True if marked successfully
        """
        return await self.cache.set(
            self.NAMESPACE_DEDUP,
            request_id,
            {"processed": True},
            ttl=ttl or self.TTL_DEDUP,
        )

    async def invalidate_llm_cache(self, model: Optional[str] = None) -> int:
        """Invalidate LLM response cache.

        Args:
            model: Model name to invalidate (all if None)

        Returns:
            Number of keys invalidated
        """
        # This would require pattern matching in Redis
        # For now, we'll just log
        logger.warning(f"LLM cache invalidation requested for model: {model or 'all'}")
        return 0

    async def invalidate_embedding_cache(self, model: Optional[str] = None) -> int:
        """Invalidate embedding cache.

        Args:
            model: Model name to invalidate (all if None)

        Returns:
            Number of keys invalidated
        """
        logger.warning(f"Embedding cache invalidation requested for model: {model or 'all'}")
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cost_saved_usd": self._cost_saved,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._cost_saved = 0.0


# Global LLM cache manager instance
_llm_cache_manager: Optional[LLMCacheManager] = None


def get_llm_cache_manager() -> LLMCacheManager:
    """Get global LLM cache manager instance.

    Returns:
        LLM cache manager
    """
    global _llm_cache_manager
    if _llm_cache_manager is None:
        _llm_cache_manager = LLMCacheManager()
    return _llm_cache_manager

