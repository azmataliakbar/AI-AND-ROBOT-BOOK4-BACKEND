# schemas.py - COMPLETE with All Required Models

from pydantic import BaseModel, Field
from typing import List, Optional


# ==================== REFERENCE MODELS ====================

class Reference(BaseModel):
    """Enhanced reference with chapter details"""
    chapter_id: Optional[str] = None
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    module: Optional[str] = None
    section: Optional[str] = None
    similarity_score: Optional[float] = None
    content_preview: Optional[str] = None


# ==================== CHAT MODELS ====================

class ChatRequest(BaseModel):
    """Chat request from frontend"""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    chapter_id: Optional[str] = Field(None, description="Optional chapter context")
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class ChatResponse(BaseModel):
    """Enhanced chat response"""
    response: str = Field(..., description="AI-generated response")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    source_type: str = Field(..., description="Source: 'qdrant', 'fallback', or 'error'")
    references: List[Reference] = Field(default=[], description="Source references")
    query_time_ms: int = Field(default=0, description="Query processing time in milliseconds")


# ==================== SEARCH MODELS ====================

class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")


class SearchResult(BaseModel):
    """Enhanced search result"""
    chapter_id: Optional[str] = None
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    module: Optional[str] = None
    content_snippet: str
    similarity_score: Optional[float] = None
    topics: List[str] = Field(default=[], description="Related topics")


class SearchResponse(BaseModel):
    """Search response"""
    results: List[SearchResult]
    total_count: int = Field(..., description="Total number of results")


# ==================== CHAPTER MODELS ====================

class ChapterSummary(BaseModel):
    """Brief chapter information"""
    id: str
    title: str
    module: int = Field(..., ge=1, le=4, description="Module number (1-4)")
    description: Optional[str] = None


class ChapterDetail(BaseModel):
    """Detailed chapter information"""
    id: str
    title: str
    module: int = Field(..., ge=1, le=4)
    content: Optional[str] = None
    description: Optional[str] = None


class ChaptersListResponse(BaseModel):
    """List of chapters response"""
    chapters: List[ChapterSummary]
    total_count: int
    offset: int = 0
    limit: int = 37


# ==================== EMBEDDING MODELS ====================

class EmbedRequest(BaseModel):
    """Request to create embeddings"""
    chapter_ids: List[str] = Field(..., min_items=1, description="List of chapter IDs to embed")
    force_rebuild: bool = Field(default=False, description="Force rebuild existing embeddings")


class EmbedResponse(BaseModel):
    """Embedding creation response"""
    processed_count: int = Field(..., description="Number of chapters processed")
    status: str = Field(..., description="Status: 'success', 'partial', or 'failed'")
    message: str = Field(..., description="Human-readable status message")


# ==================== HEALTH CHECK MODELS ====================

class ServiceStatus(BaseModel):
    """Status of a service"""
    status: str = Field(..., description="Service status")
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall status: 'healthy', 'degraded', or 'unhealthy'")
    services: dict = Field(..., description="Status of individual services")
    metadata: dict = Field(default={}, description="Additional metadata")


# ==================== STATISTICS MODELS ====================

class VectorStats(BaseModel):
    """Vector database statistics"""
    total_vectors: int
    chapters_indexed: int
    collection_name: str
    status: str


# ==================== VALIDATION MODELS ====================

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = None
    timestamp: Optional[str] = None