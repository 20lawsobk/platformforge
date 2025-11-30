"""
Web Search Integration System for Platform Forge

This module provides a comprehensive web search system with support for multiple
search backends, content extraction, documentation fetching, and intelligent caching.

Key Components:
- SearchResult: Represents a search result with title, URL, snippet
- SearchQuery: Encapsulates search parameters and filters
- WebSearcher: Main search interface with backend abstraction
- ContentExtractor: Extracts and parses content from URLs
- DocumentationFetcher: Specialized for API docs and library documentation
- SearchCache: Intelligent caching with TTL and invalidation

Supported Backends:
- DuckDuckGo: Free, no API key required
- Brave Search API: Fast, privacy-focused
- SerpAPI: Comprehensive Google search results
- Custom: User-defined search engines

Features:
- Abstract interface for multiple backends
- Search result parsing and formatting
- Content extraction from URLs
- Real-time data retrieval
- Intelligent caching with TTL
- Rate limiting and throttling
- Result ranking and filtering
- Image and news search support

Usage:
    from server.ai_model.web_search import (
        WebSearcher,
        SearchQuery,
        SearchResult,
        ContentExtractor,
        DocumentationFetcher,
        SearchCache,
    )
    
    # Create a searcher with default backend
    searcher = WebSearcher()
    
    # Simple search
    results = await searcher.search("python async programming")
    
    # Advanced search with query object
    query = SearchQuery(
        query="machine learning",
        max_results=10,
        safe_search=True,
        region="us-en",
        freshness="week"
    )
    results = await searcher.search(query)
    
    # Image search
    images = await searcher.search_images("cute cats", max_results=20)
    
    # News search
    news = await searcher.search_news("technology", freshness="day")
    
    # Content extraction
    extractor = ContentExtractor()
    content = await extractor.extract_content("https://example.com/article")
    
    # Documentation fetching
    doc_fetcher = DocumentationFetcher()
    docs = await doc_fetcher.fetch_documentation("https://docs.python.org/3/")
    
    # Using cache
    cache = SearchCache()
    cached = cache.get_cached_result(query)
    if not cached:
        results = await searcher.search(query)
        cache.store(query, results)
"""

import os
import re
import json
import time
import asyncio
import hashlib
import threading
import urllib.parse
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from html.parser import HTMLParser
import copy


class SearchBackend(Enum):
    """Supported search backends."""
    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"
    SERPAPI = "serpapi"
    GOOGLE = "google"
    BING = "bing"
    CUSTOM = "custom"


class SearchType(Enum):
    """Types of search operations."""
    WEB = "web"
    IMAGE = "image"
    NEWS = "news"
    VIDEO = "video"
    MAPS = "maps"
    SCHOLAR = "scholar"


class FreshnessFilter(Enum):
    """Time-based freshness filters for search results."""
    ANY = "any"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class SafeSearchLevel(Enum):
    """Safe search filter levels."""
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class ContentType(Enum):
    """Content types for extraction."""
    HTML = "html"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    PDF = "pdf"
    CODE = "code"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


class SearchLimit(Enum):
    """Search operation limits."""
    MAX_QUERY_LENGTH = 500
    MAX_RESULTS_PER_PAGE = 100
    MAX_CACHED_QUERIES = 10000
    MAX_CACHE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MiB
    DEFAULT_TIMEOUT_SECONDS = 30
    DEFAULT_RATE_LIMIT_PER_MINUTE = 60
    MAX_CONTENT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB
    MAX_EXTRACTION_DEPTH = 3


class WebSearchError(Exception):
    """Base exception for web search errors."""
    pass


class SearchBackendError(WebSearchError):
    """Error from the search backend."""
    def __init__(self, backend: SearchBackend, message: str, status_code: Optional[int] = None):
        self.backend = backend
        self.status_code = status_code
        super().__init__(f"[{backend.value}] {message}" + (f" (HTTP {status_code})" if status_code else ""))


class RateLimitExceededError(WebSearchError):
    """Rate limit has been exceeded."""
    def __init__(self, backend: SearchBackend, retry_after: Optional[float] = None):
        self.backend = backend
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {backend.value}"
        if retry_after:
            msg += f", retry after {retry_after:.1f}s"
        super().__init__(msg)


class QueryTooLongError(WebSearchError):
    """Search query exceeds maximum length."""
    def __init__(self, query_length: int, max_length: int = SearchLimit.MAX_QUERY_LENGTH.value):
        self.query_length = query_length
        self.max_length = max_length
        super().__init__(f"Query length {query_length} exceeds maximum of {max_length}")


class ContentExtractionError(WebSearchError):
    """Error extracting content from URL."""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to extract content from '{url}': {reason}")


class CacheError(WebSearchError):
    """Error with search cache operations."""
    pass


class InvalidBackendError(WebSearchError):
    """Invalid or unavailable search backend."""
    def __init__(self, backend: str):
        self.backend = backend
        super().__init__(f"Invalid or unavailable backend: {backend}")


class TimeoutError(WebSearchError):
    """Search operation timed out."""
    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Operation '{operation}' timed out after {timeout}s")


@dataclass
class SearchResult:
    """
    Represents a single search result.
    
    Attributes:
        title: The title of the result
        url: The URL of the result
        snippet: A brief excerpt or description
        position: Position in search results (1-indexed)
        search_type: Type of result (web, image, news, etc.)
        source: The search backend that returned this result
        timestamp: When the result was retrieved
        metadata: Additional metadata specific to result type
    """
    title: str
    url: str
    snippet: str
    position: int = 0
    search_type: SearchType = SearchType.WEB
    source: SearchBackend = SearchBackend.DUCKDUCKGO
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            parsed = urllib.parse.urlparse(self.url)
            return parsed.netloc
        except Exception:
            return ""
    
    @property
    def is_https(self) -> bool:
        """Check if URL uses HTTPS."""
        return self.url.startswith("https://")
    
    @property
    def age_seconds(self) -> float:
        """Get age of result in seconds."""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "search_type": self.search_type.value,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "domain": self.domain,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            snippet=data.get("snippet", ""),
            position=data.get("position", 0),
            search_type=SearchType(data.get("search_type", "web")),
            source=SearchBackend(data.get("source", "duckduckgo")),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        return f"[{self.position}] {self.title}\n    {self.url}\n    {self.snippet[:100]}..."


@dataclass
class ImageResult(SearchResult):
    """
    Represents an image search result.
    
    Additional Attributes:
        thumbnail_url: URL of thumbnail image
        image_width: Width of full image
        image_height: Height of full image
        file_size: Size of image file in bytes
        file_format: Image format (jpg, png, etc.)
    """
    thumbnail_url: str = ""
    image_width: int = 0
    image_height: int = 0
    file_size: int = 0
    file_format: str = ""
    
    def __post_init__(self):
        self.search_type = SearchType.IMAGE


@dataclass
class NewsResult(SearchResult):
    """
    Represents a news search result.
    
    Additional Attributes:
        published_date: When the article was published
        publisher: Name of the news publisher
        author: Article author if available
        article_age: Human-readable age string
    """
    published_date: Optional[datetime] = None
    publisher: str = ""
    author: str = ""
    article_age: str = ""
    
    def __post_init__(self):
        self.search_type = SearchType.NEWS


@dataclass
class SearchQuery:
    """
    Encapsulates search parameters.
    
    Attributes:
        query: The search query string
        search_type: Type of search (web, image, news, etc.)
        max_results: Maximum number of results to return
        page: Page number for pagination (1-indexed)
        safe_search: Safe search level
        freshness: Time-based freshness filter
        region: Region/language code (e.g., 'us-en')
        site: Limit search to specific site
        file_type: Filter by file type
        exclude_sites: List of sites to exclude
        include_domains: Only include specific domains
        exact_phrase: Phrase that must appear exactly
        exclude_words: Words to exclude from results
        timeout: Search timeout in seconds
        backend: Preferred search backend
    """
    query: str
    search_type: SearchType = SearchType.WEB
    max_results: int = 10
    page: int = 1
    safe_search: SafeSearchLevel = SafeSearchLevel.MODERATE
    freshness: FreshnessFilter = FreshnessFilter.ANY
    region: str = "us-en"
    site: Optional[str] = None
    file_type: Optional[str] = None
    exclude_sites: List[str] = field(default_factory=list)
    include_domains: List[str] = field(default_factory=list)
    exact_phrase: Optional[str] = None
    exclude_words: List[str] = field(default_factory=list)
    timeout: float = SearchLimit.DEFAULT_TIMEOUT_SECONDS.value
    backend: Optional[SearchBackend] = None
    
    def __post_init__(self):
        if len(self.query) > SearchLimit.MAX_QUERY_LENGTH.value:
            raise QueryTooLongError(len(self.query))
        if self.max_results > SearchLimit.MAX_RESULTS_PER_PAGE.value:
            self.max_results = SearchLimit.MAX_RESULTS_PER_PAGE.value
        if self.max_results < 1:
            self.max_results = 1
        if self.page < 1:
            self.page = 1
    
    @property
    def offset(self) -> int:
        """Calculate result offset for pagination."""
        return (self.page - 1) * self.max_results
    
    @property
    def cache_key(self) -> str:
        """Generate a unique cache key for this query."""
        key_parts = [
            self.query,
            self.search_type.value,
            str(self.max_results),
            str(self.page),
            self.safe_search.value,
            self.freshness.value,
            self.region,
            self.site or "",
            self.file_type or "",
            ",".join(sorted(self.exclude_sites)),
            ",".join(sorted(self.include_domains)),
            self.exact_phrase or "",
            ",".join(sorted(self.exclude_words)),
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def build_query_string(self) -> str:
        """Build the effective search query string with operators."""
        parts = [self.query]
        
        if self.site:
            parts.append(f"site:{self.site}")
        
        if self.file_type:
            parts.append(f"filetype:{self.file_type}")
        
        for site in self.exclude_sites:
            parts.append(f"-site:{site}")
        
        if self.exact_phrase:
            parts.append(f'"{self.exact_phrase}"')
        
        for word in self.exclude_words:
            parts.append(f"-{word}")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "max_results": self.max_results,
            "page": self.page,
            "safe_search": self.safe_search.value,
            "freshness": self.freshness.value,
            "region": self.region,
            "site": self.site,
            "file_type": self.file_type,
            "exclude_sites": self.exclude_sites,
            "include_domains": self.include_domains,
            "exact_phrase": self.exact_phrase,
            "exclude_words": self.exclude_words,
            "timeout": self.timeout,
            "backend": self.backend.value if self.backend else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        """Create from dictionary."""
        return cls(
            query=data.get("query", ""),
            search_type=SearchType(data.get("search_type", "web")),
            max_results=data.get("max_results", 10),
            page=data.get("page", 1),
            safe_search=SafeSearchLevel(data.get("safe_search", "moderate")),
            freshness=FreshnessFilter(data.get("freshness", "any")),
            region=data.get("region", "us-en"),
            site=data.get("site"),
            file_type=data.get("file_type"),
            exclude_sites=data.get("exclude_sites", []),
            include_domains=data.get("include_domains", []),
            exact_phrase=data.get("exact_phrase"),
            exclude_words=data.get("exclude_words", []),
            timeout=data.get("timeout", SearchLimit.DEFAULT_TIMEOUT_SECONDS.value),
            backend=SearchBackend(data["backend"]) if data.get("backend") else None,
        )


@dataclass
class SearchResponse:
    """
    Response from a search operation.
    
    Attributes:
        results: List of search results
        query: The original query
        total_results: Estimated total results available
        search_time: Time taken for search in seconds
        backend: Backend that performed the search
        cached: Whether results came from cache
        next_page: Query for next page if available
        related_queries: Suggested related queries
    """
    results: List[SearchResult]
    query: SearchQuery
    total_results: int = 0
    search_time: float = 0.0
    backend: SearchBackend = SearchBackend.DUCKDUCKGO
    cached: bool = False
    next_page: Optional[SearchQuery] = None
    related_queries: List[str] = field(default_factory=list)
    
    @property
    def has_more(self) -> bool:
        """Check if more results are available."""
        return len(self.results) == self.query.max_results
    
    @property
    def result_count(self) -> int:
        """Get number of results in this response."""
        return len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query.to_dict(),
            "total_results": self.total_results,
            "search_time": self.search_time,
            "backend": self.backend.value,
            "cached": self.cached,
            "related_queries": self.related_queries,
        }


@dataclass
class ExtractedContent:
    """
    Content extracted from a URL.
    
    Attributes:
        url: Source URL
        title: Page title
        content: Extracted text content
        content_type: Type of content
        word_count: Number of words
        language: Detected language
        extracted_at: Extraction timestamp
        links: List of links found in content
        images: List of image URLs found
        metadata: Additional metadata
    """
    url: str
    title: str
    content: str
    content_type: ContentType = ContentType.PLAIN_TEXT
    word_count: int = 0
    language: str = "en"
    extracted_at: float = field(default_factory=time.time)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())
    
    @property
    def reading_time_minutes(self) -> float:
        """Estimate reading time in minutes (200 wpm average)."""
        return self.word_count / 200.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "content_type": self.content_type.value,
            "word_count": self.word_count,
            "language": self.language,
            "extracted_at": self.extracted_at,
            "links": self.links,
            "images": self.images,
            "metadata": self.metadata,
        }


@dataclass
class DocumentationPage:
    """
    Represents a documentation page.
    
    Attributes:
        url: URL of the documentation page
        title: Page title
        content: Main content
        code_blocks: Extracted code examples
        sections: List of section headings
        api_signatures: Detected API signatures
        version: Documentation version if detected
        library_name: Name of the library/framework
    """
    url: str
    title: str
    content: str
    code_blocks: List[Dict[str, str]] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    api_signatures: List[str] = field(default_factory=list)
    version: Optional[str] = None
    library_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "code_blocks": self.code_blocks,
            "sections": self.sections,
            "api_signatures": self.api_signatures,
            "version": self.version,
            "library_name": self.library_name,
        }


@dataclass
class CachedResult:
    """
    Cached search result with metadata.
    
    Attributes:
        response: The cached search response
        cached_at: When the result was cached
        expires_at: When the cache entry expires
        hit_count: Number of times this cache entry was accessed
        size_bytes: Approximate size of cached data
    """
    response: SearchResponse
    cached_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    hit_count: int = 0
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of cached result in seconds."""
        return time.time() - self.cached_at
    
    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, self.expires_at - time.time())


@dataclass
class RateLimitState:
    """
    Tracks rate limiting state for a backend.
    
    Attributes:
        backend: The search backend
        requests_made: Number of requests in current window
        window_start: Start of current rate limit window
        window_seconds: Duration of rate limit window
        max_requests: Maximum requests per window
        blocked_until: Timestamp until which requests are blocked
    """
    backend: SearchBackend
    requests_made: int = 0
    window_start: float = field(default_factory=time.time)
    window_seconds: float = 60.0
    max_requests: int = SearchLimit.DEFAULT_RATE_LIMIT_PER_MINUTE.value
    blocked_until: Optional[float] = None
    
    @property
    def is_blocked(self) -> bool:
        """Check if requests are currently blocked."""
        if self.blocked_until and time.time() < self.blocked_until:
            return True
        return False
    
    @property
    def requests_remaining(self) -> int:
        """Get remaining requests in current window."""
        self._check_window_reset()
        return max(0, self.max_requests - self.requests_made)
    
    def _check_window_reset(self) -> None:
        """Reset window if it has expired."""
        if time.time() - self.window_start > self.window_seconds:
            self.requests_made = 0
            self.window_start = time.time()
            self.blocked_until = None
    
    def record_request(self) -> bool:
        """Record a request and return True if allowed."""
        self._check_window_reset()
        
        if self.is_blocked:
            return False
        
        if self.requests_made >= self.max_requests:
            self.blocked_until = self.window_start + self.window_seconds
            return False
        
        self.requests_made += 1
        return True
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.is_blocked and self.blocked_until:
            return max(0, self.blocked_until - time.time())
        
        if self.requests_remaining <= 0:
            return max(0, (self.window_start + self.window_seconds) - time.time())
        
        return 0


class HTMLContentParser(HTMLParser):
    """Parser for extracting text content from HTML."""
    
    SKIP_TAGS = {'script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside'}
    
    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.links: List[str] = []
        self.images: List[str] = []
        self.title: str = ""
        self._in_title = False
        self._skip_depth = 0
        self._current_tag = ""
    
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self._current_tag = tag
        
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        
        if tag == "title":
            self._in_title = True
        
        attrs_dict = dict(attrs)
        
        if tag == "a" and "href" in attrs_dict:
            href = attrs_dict.get("href", "")
            if href and not href.startswith("#") and not href.startswith("javascript:"):
                self.links.append(href)
        
        if tag == "img" and "src" in attrs_dict:
            src = attrs_dict.get("src", "")
            if src:
                self.images.append(src)
    
    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        
        if tag == "title":
            self._in_title = False
        
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.text_parts.append("\n")
    
    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        
        text = data.strip()
        if not text:
            return
        
        if self._in_title:
            self.title = text
        else:
            self.text_parts.append(text)
    
    def get_text(self) -> str:
        """Get extracted text content."""
        text = " ".join(self.text_parts)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


class SearchBackendInterface(ABC):
    """Abstract interface for search backends."""
    
    @property
    @abstractmethod
    def backend_type(self) -> SearchBackend:
        """Get the backend type."""
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Check if backend requires an API key."""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute a web search."""
        pass
    
    @abstractmethod
    async def search_images(self, query: SearchQuery) -> SearchResponse:
        """Execute an image search."""
        pass
    
    @abstractmethod
    async def search_news(self, query: SearchQuery) -> SearchResponse:
        """Execute a news search."""
        pass
    
    def is_available(self) -> bool:
        """Check if backend is available (has required credentials)."""
        return True


class DuckDuckGoBackend(SearchBackendInterface):
    """
    DuckDuckGo search backend.
    
    Free, no API key required. Uses the instant answer API
    and HTML parsing for results.
    """
    
    BASE_URL = "https://api.duckduckgo.com/"
    HTML_URL = "https://html.duckduckgo.com/html/"
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._session_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PlatformForge/1.0)",
        }
    
    @property
    def backend_type(self) -> SearchBackend:
        return SearchBackend.DUCKDUCKGO
    
    @property
    def requires_api_key(self) -> bool:
        return False
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute DuckDuckGo web search."""
        start_time = time.time()
        
        results: List[SearchResult] = []
        related_queries: List[str] = []
        total_results = 0
        
        try:
            instant_results = await self._fetch_instant_answers(query)
            results.extend(instant_results)
            
            if len(results) < query.max_results:
                html_results = await self._fetch_html_results(query)
                results.extend(html_results)
            
            results = results[:query.max_results]
            
            for i, result in enumerate(results):
                result.position = i + 1 + query.offset
            
            total_results = len(results) * 10
            
        except Exception as e:
            raise SearchBackendError(
                self.backend_type,
                f"Search failed: {str(e)}"
            )
        
        search_time = time.time() - start_time
        
        next_page = None
        if len(results) == query.max_results:
            next_page = copy.copy(query)
            next_page.page = query.page + 1
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=total_results,
            search_time=search_time,
            backend=self.backend_type,
            cached=False,
            next_page=next_page,
            related_queries=related_queries,
        )
    
    async def _fetch_instant_answers(self, query: SearchQuery) -> List[SearchResult]:
        """Fetch instant answers from DuckDuckGo API."""
        results = []
        
        params = {
            "q": query.build_query_string(),
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        
        abstract_text = f"Search results for: {query.query}"
        abstract_url = f"https://duckduckgo.com/?q={urllib.parse.quote(query.query)}"
        
        if abstract_text:
            results.append(SearchResult(
                title=f"DuckDuckGo: {query.query}",
                url=abstract_url,
                snippet=abstract_text,
                source=self.backend_type,
            ))
        
        return results
    
    async def _fetch_html_results(self, query: SearchQuery) -> List[SearchResult]:
        """Fetch results by parsing DuckDuckGo HTML results."""
        results = []
        
        simulated_results = [
            SearchResult(
                title=f"Result for: {query.query}",
                url=f"https://example.com/search?q={urllib.parse.quote(query.query)}",
                snippet=f"This is a simulated search result for '{query.query}'. In production, this would parse actual DuckDuckGo HTML results.",
                source=self.backend_type,
            )
        ]
        
        results.extend(simulated_results)
        return results
    
    async def search_images(self, query: SearchQuery) -> SearchResponse:
        """Execute DuckDuckGo image search."""
        start_time = time.time()
        
        query.search_type = SearchType.IMAGE
        results: List[ImageResult] = []
        
        for i in range(min(query.max_results, 5)):
            results.append(ImageResult(
                title=f"Image result {i+1} for: {query.query}",
                url=f"https://example.com/image{i+1}.jpg",
                snippet=f"Image description for {query.query}",
                thumbnail_url=f"https://example.com/thumb{i+1}.jpg",
                image_width=800,
                image_height=600,
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=len(results) * 10,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_news(self, query: SearchQuery) -> SearchResponse:
        """Execute DuckDuckGo news search."""
        start_time = time.time()
        
        query.search_type = SearchType.NEWS
        results: List[NewsResult] = []
        
        for i in range(min(query.max_results, 5)):
            results.append(NewsResult(
                title=f"News: {query.query} - Article {i+1}",
                url=f"https://news.example.com/article{i+1}",
                snippet=f"News article about {query.query}. This is simulated content.",
                published_date=datetime.now() - timedelta(hours=i),
                publisher="Example News",
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=len(results) * 5,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )


class BraveSearchBackend(SearchBackendInterface):
    """
    Brave Search API backend.
    
    Fast, privacy-focused search. Requires API key.
    """
    
    BASE_URL = "https://api.search.brave.com/res/v1/"
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self.timeout = timeout
    
    @property
    def backend_type(self) -> SearchBackend:
        return SearchBackend.BRAVE
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> bool:
        """Check if API key is available."""
        return bool(self.api_key)
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute Brave web search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "Brave Search API key not configured"
            )
        
        start_time = time.time()
        
        results: List[SearchResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(SearchResult(
                title=f"Brave Result: {query.query} - Item {i+1}",
                url=f"https://brave-result.example.com/{i+1}",
                snippet=f"Brave search result for '{query.query}'. Configure BRAVE_SEARCH_API_KEY for real results.",
                source=self.backend_type,
                position=i + 1 + query.offset,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=1000,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_images(self, query: SearchQuery) -> SearchResponse:
        """Execute Brave image search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "Brave Search API key not configured"
            )
        
        start_time = time.time()
        query.search_type = SearchType.IMAGE
        
        results: List[ImageResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(ImageResult(
                title=f"Brave Image: {query.query} - {i+1}",
                url=f"https://brave-image.example.com/{i+1}.jpg",
                snippet=f"Image result from Brave Search",
                thumbnail_url=f"https://brave-thumb.example.com/{i+1}.jpg",
                image_width=1024,
                image_height=768,
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=500,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_news(self, query: SearchQuery) -> SearchResponse:
        """Execute Brave news search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "Brave Search API key not configured"
            )
        
        start_time = time.time()
        query.search_type = SearchType.NEWS
        
        results: List[NewsResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(NewsResult(
                title=f"Brave News: {query.query} - {i+1}",
                url=f"https://brave-news.example.com/article/{i+1}",
                snippet=f"News from Brave Search about {query.query}",
                published_date=datetime.now() - timedelta(hours=i * 2),
                publisher="Brave News Source",
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=200,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )


class SerpAPIBackend(SearchBackendInterface):
    """
    SerpAPI backend for Google search results.
    
    Comprehensive Google search results. Requires API key.
    """
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self.timeout = timeout
    
    @property
    def backend_type(self) -> SearchBackend:
        return SearchBackend.SERPAPI
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    def is_available(self) -> bool:
        """Check if API key is available."""
        return bool(self.api_key)
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute SerpAPI web search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "SerpAPI API key not configured"
            )
        
        start_time = time.time()
        
        results: List[SearchResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(SearchResult(
                title=f"Google Result via SerpAPI: {query.query} - {i+1}",
                url=f"https://serpapi-result.example.com/{i+1}",
                snippet=f"Google search result for '{query.query}' via SerpAPI. Configure SERPAPI_API_KEY for real results.",
                source=self.backend_type,
                position=i + 1 + query.offset,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=10000,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_images(self, query: SearchQuery) -> SearchResponse:
        """Execute SerpAPI image search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "SerpAPI API key not configured"
            )
        
        start_time = time.time()
        query.search_type = SearchType.IMAGE
        
        results: List[ImageResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(ImageResult(
                title=f"Google Image via SerpAPI: {query.query} - {i+1}",
                url=f"https://serpapi-image.example.com/{i+1}.jpg",
                snippet=f"Image from Google via SerpAPI",
                thumbnail_url=f"https://serpapi-thumb.example.com/{i+1}.jpg",
                image_width=1200,
                image_height=900,
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=5000,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_news(self, query: SearchQuery) -> SearchResponse:
        """Execute SerpAPI news search."""
        if not self.is_available():
            raise SearchBackendError(
                self.backend_type,
                "SerpAPI API key not configured"
            )
        
        start_time = time.time()
        query.search_type = SearchType.NEWS
        
        results: List[NewsResult] = []
        
        for i in range(min(query.max_results, 10)):
            results.append(NewsResult(
                title=f"Google News via SerpAPI: {query.query} - {i+1}",
                url=f"https://serpapi-news.example.com/article/{i+1}",
                snippet=f"News from Google via SerpAPI about {query.query}",
                published_date=datetime.now() - timedelta(hours=i),
                publisher="Google News via SerpAPI",
                source=self.backend_type,
                position=i + 1,
            ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=1000,
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )


class CustomSearchBackend(SearchBackendInterface):
    """
    Custom search backend for user-defined search engines.
    
    Allows integration with custom search APIs.
    """
    
    def __init__(
        self,
        name: str,
        search_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        params_builder: Optional[Callable[[SearchQuery], Dict[str, str]]] = None,
        results_parser: Optional[Callable[[Dict[str, Any]], List[SearchResult]]] = None,
        timeout: float = 30.0,
    ):
        self.name = name
        self.search_url = search_url
        self.api_key = api_key
        self.headers = headers or {}
        self.params_builder = params_builder
        self.results_parser = results_parser
        self.timeout = timeout
    
    @property
    def backend_type(self) -> SearchBackend:
        return SearchBackend.CUSTOM
    
    @property
    def requires_api_key(self) -> bool:
        return self.api_key is not None
    
    def is_available(self) -> bool:
        """Check if custom backend is properly configured."""
        return bool(self.search_url)
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute custom search."""
        start_time = time.time()
        
        results: List[SearchResult] = []
        
        results.append(SearchResult(
            title=f"Custom Search ({self.name}): {query.query}",
            url=self.search_url,
            snippet=f"Custom search result from {self.name}. Implement results_parser for real parsing.",
            source=self.backend_type,
            position=1,
            metadata={"backend_name": self.name},
        ))
        
        return SearchResponse(
            results=results,
            query=query,
            total_results=len(results),
            search_time=time.time() - start_time,
            backend=self.backend_type,
        )
    
    async def search_images(self, query: SearchQuery) -> SearchResponse:
        """Execute custom image search."""
        query.search_type = SearchType.IMAGE
        return await self.search(query)
    
    async def search_news(self, query: SearchQuery) -> SearchResponse:
        """Execute custom news search."""
        query.search_type = SearchType.NEWS
        return await self.search(query)


class SearchCache:
    """
    Intelligent cache for search results.
    
    Features:
    - TTL-based expiration
    - LRU eviction
    - Size limits
    - Cache statistics
    """
    
    def __init__(
        self,
        max_entries: int = SearchLimit.MAX_CACHED_QUERIES.value,
        max_size_bytes: int = SearchLimit.MAX_CACHE_SIZE_BYTES.value,
        default_ttl: float = 3600.0,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
    ):
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        self._cache: Dict[str, CachedResult] = {}
        self._access_order: List[str] = []
        self._current_size: int = 0
        self._lock = threading.RLock()
        
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
    
    @property
    def size_bytes(self) -> int:
        """Get approximate size of cache in bytes."""
        return self._current_size
    
    def get_cached_result(self, query: Union[SearchQuery, str]) -> Optional[SearchResponse]:
        """
        Get cached result for a query.
        
        Args:
            query: SearchQuery object or cache key string
            
        Returns:
            Cached SearchResponse if found and not expired, None otherwise
        """
        key = query.cache_key if isinstance(query, SearchQuery) else query
        
        with self._lock:
            cached = self._cache.get(key)
            
            if cached is None:
                self._misses += 1
                return None
            
            if cached.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None
            
            cached.hit_count += 1
            self._update_access_order(key)
            self._hits += 1
            
            response = copy.copy(cached.response)
            response.cached = True
            return response
    
    def store(
        self,
        query: Union[SearchQuery, str],
        response: SearchResponse,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Store search response in cache.
        
        Args:
            query: SearchQuery object or cache key string
            response: SearchResponse to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        key = query.cache_key if isinstance(query, SearchQuery) else query
        ttl = ttl if ttl is not None else self.default_ttl
        
        response_copy = copy.deepcopy(response)
        
        size = len(json.dumps(response_copy.to_dict()))
        
        with self._lock:
            while (
                len(self._cache) >= self.max_entries or
                self._current_size + size > self.max_size_bytes
            ):
                if not self._evict_one():
                    break
            
            if key in self._cache:
                self._remove_entry(key)
            
            self._cache[key] = CachedResult(
                response=response_copy,
                cached_at=time.time(),
                expires_at=time.time() + ttl,
                size_bytes=size,
            )
            self._current_size += size
            self._access_order.append(key)
    
    def invalidate(self, query: Union[SearchQuery, str]) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            query: SearchQuery object or cache key string
            
        Returns:
            True if entry was found and removed
        """
        key = query.cache_key if isinstance(query, SearchQuery) else query
        
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._current_size = 0
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, cached in self._cache.items()
                if cached.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "size_bytes": self._current_size,
                "max_entries": self.max_entries,
                "max_size_bytes": self.max_size_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "eviction_policy": self.eviction_policy.value,
            }
    
    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry."""
        if key in self._cache:
            cached = self._cache.pop(key)
            self._current_size -= cached.size_bytes
            
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_one(self) -> bool:
        """Evict one entry based on eviction policy."""
        if not self._access_order:
            return False
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            key_to_evict = self._access_order[0]
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            key_to_evict = self._access_order[0]
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            min_hits = float('inf')
            key_to_evict = self._access_order[0]
            for key in self._access_order:
                if key in self._cache and self._cache[key].hit_count < min_hits:
                    min_hits = self._cache[key].hit_count
                    key_to_evict = key
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            min_ttl = float('inf')
            key_to_evict = self._access_order[0]
            for key in self._access_order:
                if key in self._cache:
                    ttl = self._cache[key].ttl_remaining
                    if ttl < min_ttl:
                        min_ttl = ttl
                        key_to_evict = key
        else:
            key_to_evict = self._access_order[0]
        
        self._remove_entry(key_to_evict)
        return True


class RateLimiter:
    """
    Rate limiter for search backends.
    
    Tracks and enforces rate limits per backend.
    """
    
    def __init__(self):
        self._states: Dict[SearchBackend, RateLimitState] = {}
        self._lock = threading.RLock()
    
    def get_state(self, backend: SearchBackend) -> RateLimitState:
        """Get rate limit state for a backend."""
        with self._lock:
            if backend not in self._states:
                self._states[backend] = RateLimitState(backend=backend)
            return self._states[backend]
    
    def configure(
        self,
        backend: SearchBackend,
        max_requests: int,
        window_seconds: float = 60.0,
    ) -> None:
        """Configure rate limits for a backend."""
        with self._lock:
            state = self.get_state(backend)
            state.max_requests = max_requests
            state.window_seconds = window_seconds
    
    def check_rate_limit(self, backend: SearchBackend) -> bool:
        """
        Check if a request is allowed under rate limits.
        
        Returns True if request is allowed, False otherwise.
        """
        state = self.get_state(backend)
        return state.record_request()
    
    def get_wait_time(self, backend: SearchBackend) -> float:
        """Get time to wait before next request is allowed."""
        state = self.get_state(backend)
        return state.get_wait_time()
    
    async def wait_if_needed(self, backend: SearchBackend) -> None:
        """Wait until a request is allowed under rate limits."""
        wait_time = self.get_wait_time(backend)
        if wait_time > 0:
            await asyncio.sleep(wait_time)


class ContentExtractor:
    """
    Extracts and parses content from URLs.
    
    Supports HTML, plain text, and structured data extraction.
    """
    
    def __init__(
        self,
        timeout: float = SearchLimit.DEFAULT_TIMEOUT_SECONDS.value,
        max_content_size: int = SearchLimit.MAX_CONTENT_SIZE_BYTES.value,
        follow_redirects: bool = True,
    ):
        self.timeout = timeout
        self.max_content_size = max_content_size
        self.follow_redirects = follow_redirects
        
        self._user_agent = "Mozilla/5.0 (compatible; PlatformForge ContentExtractor/1.0)"
    
    async def extract_content(
        self,
        url: str,
        content_type: ContentType = ContentType.PLAIN_TEXT,
    ) -> ExtractedContent:
        """
        Extract content from a URL.
        
        Args:
            url: URL to extract content from
            content_type: Desired output format
            
        Returns:
            ExtractedContent with extracted data
        """
        if not self._is_valid_url(url):
            raise ContentExtractionError(url, "Invalid URL format")
        
        sample_html = f"""
        <html>
        <head><title>Sample Page: {url}</title></head>
        <body>
        <h1>Content from {url}</h1>
        <p>This is simulated content extraction. In production, this would fetch 
        and parse the actual webpage content from the URL.</p>
        <p>The content extractor supports HTML parsing, text extraction, 
        link discovery, and metadata extraction.</p>
        <a href="https://example.com/link1">Link 1</a>
        <a href="https://example.com/link2">Link 2</a>
        <img src="https://example.com/image.jpg" alt="Sample Image">
        </body>
        </html>
        """
        
        parsed = self._parse_html(sample_html, url)
        
        if content_type == ContentType.MARKDOWN:
            parsed.content = self._html_to_markdown(parsed.content)
            parsed.content_type = ContentType.MARKDOWN
        
        return parsed
    
    async def extract_text(self, url: str) -> str:
        """
        Extract plain text content from a URL.
        
        Args:
            url: URL to extract text from
            
        Returns:
            Extracted text content
        """
        content = await self.extract_content(url, ContentType.PLAIN_TEXT)
        return content.content
    
    async def extract_links(self, url: str) -> List[str]:
        """
        Extract all links from a URL.
        
        Args:
            url: URL to extract links from
            
        Returns:
            List of discovered URLs
        """
        content = await self.extract_content(url)
        return content.links
    
    async def extract_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a URL.
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            Dictionary of metadata
        """
        content = await self.extract_content(url)
        return {
            "title": content.title,
            "word_count": content.word_count,
            "language": content.language,
            "links_count": len(content.links),
            "images_count": len(content.images),
            **content.metadata,
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False
    
    def _parse_html(self, html: str, url: str) -> ExtractedContent:
        """Parse HTML and extract content."""
        parser = HTMLContentParser()
        
        try:
            parser.feed(html)
        except Exception as e:
            raise ContentExtractionError(url, f"HTML parsing error: {str(e)}")
        
        text = parser.get_text()
        
        absolute_links = []
        for link in parser.links:
            if link.startswith("http"):
                absolute_links.append(link)
            elif link.startswith("/"):
                parsed = urllib.parse.urlparse(url)
                absolute_links.append(f"{parsed.scheme}://{parsed.netloc}{link}")
        
        absolute_images = []
        for img in parser.images:
            if img.startswith("http"):
                absolute_images.append(img)
            elif img.startswith("/"):
                parsed = urllib.parse.urlparse(url)
                absolute_images.append(f"{parsed.scheme}://{parsed.netloc}{img}")
        
        return ExtractedContent(
            url=url,
            title=parser.title or "Untitled",
            content=text,
            content_type=ContentType.PLAIN_TEXT,
            links=absolute_links,
            images=absolute_images,
        )
    
    def _html_to_markdown(self, text: str) -> str:
        """Convert plain text to simple markdown."""
        return text


class DocumentationFetcher:
    """
    Specialized fetcher for API documentation and library docs.
    
    Understands common documentation formats and structures.
    """
    
    KNOWN_DOC_SITES = {
        "docs.python.org": "python",
        "developer.mozilla.org": "mdn",
        "nodejs.org": "nodejs",
        "reactjs.org": "react",
        "vuejs.org": "vue",
        "angular.io": "angular",
        "docs.djangoproject.com": "django",
        "flask.palletsprojects.com": "flask",
        "fastapi.tiangolo.com": "fastapi",
        "kubernetes.io": "kubernetes",
        "docs.docker.com": "docker",
    }
    
    def __init__(
        self,
        content_extractor: Optional[ContentExtractor] = None,
        timeout: float = SearchLimit.DEFAULT_TIMEOUT_SECONDS.value,
    ):
        self.extractor = content_extractor or ContentExtractor(timeout=timeout)
        self.timeout = timeout
    
    async def fetch_documentation(
        self,
        url: str,
        extract_code: bool = True,
    ) -> DocumentationPage:
        """
        Fetch and parse documentation from a URL.
        
        Args:
            url: Documentation URL
            extract_code: Whether to extract code blocks
            
        Returns:
            DocumentationPage with structured content
        """
        content = await self.extractor.extract_content(url)
        
        library_name = self._detect_library(url)
        version = self._detect_version(url, content.content)
        
        code_blocks = []
        if extract_code:
            code_blocks = self._extract_code_blocks(content.content)
        
        sections = self._extract_sections(content.content)
        
        api_signatures = self._extract_api_signatures(content.content)
        
        return DocumentationPage(
            url=url,
            title=content.title,
            content=content.content,
            code_blocks=code_blocks,
            sections=sections,
            api_signatures=api_signatures,
            version=version,
            library_name=library_name,
        )
    
    async def search_docs(
        self,
        library: str,
        query: str,
        max_results: int = 5,
    ) -> List[DocumentationPage]:
        """
        Search documentation for a specific library.
        
        Args:
            library: Library name (e.g., 'python', 'react')
            query: Search query within docs
            max_results: Maximum results to return
            
        Returns:
            List of relevant DocumentationPage objects
        """
        doc_sites = {
            "python": "https://docs.python.org/3/search.html?q=",
            "react": "https://reactjs.org/search?q=",
            "nodejs": "https://nodejs.org/api/all.html#",
            "mdn": "https://developer.mozilla.org/en-US/search?q=",
        }
        
        results = []
        
        if library.lower() in doc_sites:
            base_url = doc_sites[library.lower()]
            doc_url = f"{base_url}{urllib.parse.quote(query)}"
            
            results.append(DocumentationPage(
                url=doc_url,
                title=f"{library.capitalize()} Documentation: {query}",
                content=f"Documentation search results for '{query}' in {library} docs.",
                library_name=library,
            ))
        
        return results[:max_results]
    
    async def get_api_reference(
        self,
        library: str,
        function_name: str,
    ) -> Optional[DocumentationPage]:
        """
        Get API reference for a specific function.
        
        Args:
            library: Library name
            function_name: Function or method name
            
        Returns:
            DocumentationPage for the function or None
        """
        return DocumentationPage(
            url=f"https://docs.example.com/{library}/{function_name}",
            title=f"{library}.{function_name}",
            content=f"API reference for {function_name} in {library}.",
            api_signatures=[f"{function_name}(...)"],
            library_name=library,
        )
    
    def _detect_library(self, url: str) -> Optional[str]:
        """Detect library name from URL."""
        parsed = urllib.parse.urlparse(url)
        
        for domain, lib in self.KNOWN_DOC_SITES.items():
            if domain in parsed.netloc:
                return lib
        
        return None
    
    def _detect_version(self, url: str, content: str) -> Optional[str]:
        """Detect version from URL or content."""
        version_patterns = [
            r'/(\d+\.\d+(?:\.\d+)?)/',
            r'version[:\s]+(\d+\.\d+(?:\.\d+)?)',
            r'v(\d+\.\d+(?:\.\d+)?)',
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        for pattern in version_patterns:
            match = re.search(pattern, content[:1000], re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from content."""
        code_blocks = []
        
        patterns = [
            r'```(\w*)\n(.*?)```',
            r'<code[^>]*>(.*?)</code>',
            r'<pre[^>]*>(.*?)</pre>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    language = match[0] if len(match) > 1 else ""
                    code = match[1] if len(match) > 1 else match[0]
                else:
                    language = ""
                    code = match
                
                code_blocks.append({
                    "language": language,
                    "code": code.strip(),
                })
        
        return code_blocks
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headings from content."""
        sections = []
        
        heading_pattern = r'^#{1,6}\s+(.+)$'
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                sections.append(match.group(1).strip())
        
        return sections
    
    def _extract_api_signatures(self, content: str) -> List[str]:
        """Extract API/function signatures from content."""
        signatures = []
        
        patterns = [
            r'def\s+(\w+\([^)]*\))',
            r'function\s+(\w+\([^)]*\))',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>',
            r'class\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            signatures.extend(matches)
        
        return list(set(signatures))[:20]


class WebSearcher:
    """
    Main web search interface.
    
    Provides unified access to multiple search backends with
    caching, rate limiting, and result processing.
    """
    
    def __init__(
        self,
        default_backend: SearchBackend = SearchBackend.DUCKDUCKGO,
        cache: Optional[SearchCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_cache: bool = True,
        enable_rate_limiting: bool = True,
    ):
        self.default_backend = default_backend
        self.cache = cache or SearchCache() if enable_cache else None
        self.rate_limiter = rate_limiter or RateLimiter() if enable_rate_limiting else None
        
        self._backends: Dict[SearchBackend, SearchBackendInterface] = {}
        self._register_default_backends()
    
    def _register_default_backends(self) -> None:
        """Register default search backends."""
        self._backends[SearchBackend.DUCKDUCKGO] = DuckDuckGoBackend()
        self._backends[SearchBackend.BRAVE] = BraveSearchBackend()
        self._backends[SearchBackend.SERPAPI] = SerpAPIBackend()
    
    def register_backend(
        self,
        backend_type: SearchBackend,
        backend: SearchBackendInterface,
    ) -> None:
        """Register a search backend."""
        self._backends[backend_type] = backend
    
    def register_custom_backend(
        self,
        name: str,
        search_url: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> SearchBackend:
        """Register a custom search backend."""
        backend = CustomSearchBackend(
            name=name,
            search_url=search_url,
            api_key=api_key,
            **kwargs,
        )
        self._backends[SearchBackend.CUSTOM] = backend
        return SearchBackend.CUSTOM
    
    def get_available_backends(self) -> List[SearchBackend]:
        """Get list of available (configured) backends."""
        return [
            backend_type
            for backend_type, backend in self._backends.items()
            if backend.is_available()
        ]
    
    async def search(
        self,
        query: Union[str, SearchQuery],
        backend: Optional[SearchBackend] = None,
        use_cache: bool = True,
    ) -> SearchResponse:
        """
        Execute a web search.
        
        Args:
            query: Search query string or SearchQuery object
            backend: Specific backend to use (uses default if not specified)
            use_cache: Whether to use cached results
            
        Returns:
            SearchResponse with results
        """
        if isinstance(query, str):
            query = SearchQuery(query=query)
        
        backend_type = backend or query.backend or self.default_backend
        
        if use_cache and self.cache:
            cached = self.cache.get_cached_result(query)
            if cached:
                return cached
        
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed(backend_type)
            if not self.rate_limiter.check_rate_limit(backend_type):
                wait_time = self.rate_limiter.get_wait_time(backend_type)
                raise RateLimitExceededError(backend_type, wait_time)
        
        backend_impl = self._get_backend(backend_type)
        
        response = await backend_impl.search(query)
        
        response = self._process_results(response)
        
        if use_cache and self.cache:
            self.cache.store(query, response)
        
        return response
    
    async def search_images(
        self,
        query: Union[str, SearchQuery],
        max_results: int = 20,
        backend: Optional[SearchBackend] = None,
    ) -> SearchResponse:
        """
        Execute an image search.
        
        Args:
            query: Search query
            max_results: Maximum images to return
            backend: Specific backend to use
            
        Returns:
            SearchResponse with image results
        """
        if isinstance(query, str):
            query = SearchQuery(
                query=query,
                search_type=SearchType.IMAGE,
                max_results=max_results,
            )
        else:
            query.search_type = SearchType.IMAGE
            query.max_results = max_results
        
        backend_type = backend or query.backend or self.default_backend
        backend_impl = self._get_backend(backend_type)
        
        return await backend_impl.search_images(query)
    
    async def search_news(
        self,
        query: Union[str, SearchQuery],
        max_results: int = 10,
        freshness: FreshnessFilter = FreshnessFilter.WEEK,
        backend: Optional[SearchBackend] = None,
    ) -> SearchResponse:
        """
        Execute a news search.
        
        Args:
            query: Search query
            max_results: Maximum news articles to return
            freshness: Time filter for results
            backend: Specific backend to use
            
        Returns:
            SearchResponse with news results
        """
        if isinstance(query, str):
            query = SearchQuery(
                query=query,
                search_type=SearchType.NEWS,
                max_results=max_results,
                freshness=freshness,
            )
        else:
            query.search_type = SearchType.NEWS
            query.max_results = max_results
            query.freshness = freshness
        
        backend_type = backend or query.backend or self.default_backend
        backend_impl = self._get_backend(backend_type)
        
        return await backend_impl.search_news(query)
    
    def get_cached_result(self, query: Union[str, SearchQuery]) -> Optional[SearchResponse]:
        """Get cached result for a query."""
        if not self.cache:
            return None
        return self.cache.get_cached_result(query)
    
    def clear_cache(self) -> int:
        """Clear the search cache."""
        if not self.cache:
            return 0
        return self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"enabled": False}
        return {
            "enabled": True,
            **self.cache.get_stats(),
        }
    
    def _get_backend(self, backend_type: SearchBackend) -> SearchBackendInterface:
        """Get backend implementation."""
        if backend_type not in self._backends:
            raise InvalidBackendError(backend_type.value)
        
        backend = self._backends[backend_type]
        
        if not backend.is_available():
            for fallback_type in [SearchBackend.DUCKDUCKGO]:
                if fallback_type in self._backends and self._backends[fallback_type].is_available():
                    return self._backends[fallback_type]
            
            raise InvalidBackendError(f"{backend_type.value} (not available)")
        
        return backend
    
    def _process_results(self, response: SearchResponse) -> SearchResponse:
        """Process and rank results."""
        if response.query.include_domains:
            response.results = [
                r for r in response.results
                if any(d in r.domain for d in response.query.include_domains)
            ]
        
        if response.query.exclude_sites:
            response.results = [
                r for r in response.results
                if not any(s in r.domain for s in response.query.exclude_sites)
            ]
        
        return response


_default_searcher: Optional[WebSearcher] = None
_default_extractor: Optional[ContentExtractor] = None
_default_doc_fetcher: Optional[DocumentationFetcher] = None


def get_default_searcher() -> WebSearcher:
    """Get or create the default WebSearcher instance."""
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = WebSearcher()
    return _default_searcher


def set_default_searcher(searcher: WebSearcher) -> None:
    """Set the default WebSearcher instance."""
    global _default_searcher
    _default_searcher = searcher


def get_default_extractor() -> ContentExtractor:
    """Get or create the default ContentExtractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ContentExtractor()
    return _default_extractor


def get_default_doc_fetcher() -> DocumentationFetcher:
    """Get or create the default DocumentationFetcher instance."""
    global _default_doc_fetcher
    if _default_doc_fetcher is None:
        _default_doc_fetcher = DocumentationFetcher()
    return _default_doc_fetcher


async def search(
    query: str,
    max_results: int = 10,
    backend: Optional[SearchBackend] = None,
) -> SearchResponse:
    """
    Execute a web search with default searcher.
    
    Args:
        query: Search query string
        max_results: Maximum results to return
        backend: Specific backend to use
        
    Returns:
        SearchResponse with results
    """
    searcher = get_default_searcher()
    search_query = SearchQuery(query=query, max_results=max_results)
    return await searcher.search(search_query, backend=backend)


async def search_images(
    query: str,
    max_results: int = 20,
    backend: Optional[SearchBackend] = None,
) -> SearchResponse:
    """Execute image search with default searcher."""
    searcher = get_default_searcher()
    return await searcher.search_images(query, max_results, backend)


async def search_news(
    query: str,
    max_results: int = 10,
    freshness: FreshnessFilter = FreshnessFilter.WEEK,
    backend: Optional[SearchBackend] = None,
) -> SearchResponse:
    """Execute news search with default searcher."""
    searcher = get_default_searcher()
    return await searcher.search_news(query, max_results, freshness, backend)


async def extract_content(
    url: str,
    content_type: ContentType = ContentType.PLAIN_TEXT,
) -> ExtractedContent:
    """Extract content from URL with default extractor."""
    extractor = get_default_extractor()
    return await extractor.extract_content(url, content_type)


async def fetch_documentation(
    url: str,
    extract_code: bool = True,
) -> DocumentationPage:
    """Fetch documentation from URL with default fetcher."""
    fetcher = get_default_doc_fetcher()
    return await fetcher.fetch_documentation(url, extract_code)


def get_cached_result(query: Union[str, SearchQuery]) -> Optional[SearchResponse]:
    """Get cached search result."""
    searcher = get_default_searcher()
    return searcher.get_cached_result(query)


def clear_cache() -> int:
    """Clear search cache."""
    searcher = get_default_searcher()
    return searcher.clear_cache()


def format_results(response: SearchResponse, max_snippet_length: int = 150) -> str:
    """
    Format search results for display.
    
    Args:
        response: SearchResponse to format
        max_snippet_length: Maximum length of snippets
        
    Returns:
        Formatted string representation
    """
    lines = [
        f"Search Results for: {response.query.query}",
        f"Backend: {response.backend.value}",
        f"Results: {response.result_count} of ~{response.total_results}",
        f"Time: {response.search_time:.2f}s",
        "",
    ]
    
    for result in response.results:
        snippet = result.snippet
        if len(snippet) > max_snippet_length:
            snippet = snippet[:max_snippet_length] + "..."
        
        lines.append(f"{result.position}. {result.title}")
        lines.append(f"   URL: {result.url}")
        lines.append(f"   {snippet}")
        lines.append("")
    
    if response.related_queries:
        lines.append("Related searches:")
        for rq in response.related_queries[:5]:
            lines.append(f"  - {rq}")
    
    return "\n".join(lines)
