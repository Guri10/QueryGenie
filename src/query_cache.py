"""
Query Result Caching for QueryGenie
Implements LRU cache for frequently asked queries
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass, asdict


@dataclass
class CachedResult:
    """Cached query result"""
    query: str
    result: Dict[str, Any]
    timestamp: float
    hit_count: int = 0


class QueryCache:
    """
    LRU Cache for query results
    
    Significantly speeds up repeated or similar queries
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize query cache
        
        Args:
            max_size: Maximum number of cached queries
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CachedResult] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _hash_query(self, query: str, k: int) -> str:
        """Create hash key for query"""
        key = f"{query.lower().strip()}_{k}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, query: str, k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available
        
        Args:
            query: User query
            k: Number of results
            
        Returns:
            Cached result or None
        """
        key = self._hash_query(query, k)
        
        if key in self.cache:
            cached = self.cache[key]
            
            # Check if expired
            if time.time() - cached.timestamp > self.ttl:
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            cached.hit_count += 1
            self.stats["hits"] += 1
            
            return cached.result
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, k: int, result: Dict[str, Any]):
        """
        Cache query result
        
        Args:
            query: User query
            k: Number of results
            result: Query result to cache
        """
        key = self._hash_query(query, k)
        
        # If at max size, remove oldest
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
        
        self.cache[key] = CachedResult(
            query=query,
            result=result,
            timestamp=time.time(),
            hit_count=0
        )
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "ttl": self.ttl
        }
    
    def get_top_queries(self, n: int = 10) -> list:
        """Get most frequently cached queries"""
        sorted_cache = sorted(
            self.cache.values(),
            key=lambda x: x.hit_count,
            reverse=True
        )
        return [
            {"query": c.query, "hits": c.hit_count}
            for c in sorted_cache[:n]
        ]

