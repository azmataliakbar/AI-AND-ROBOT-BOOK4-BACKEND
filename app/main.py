# main.py - ENHANCED VERSION with Better Error Handling

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from .database import engine, get_db, SessionLocal
from . import models, schemas
from .rag import rag_service
from .embeddings import embedding_service
from .logging_config import logger as structured_logger
from .vector_db import vector_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Chatbot API",
    description="RAG-powered chatbot for the Physical AI textbook",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Enhanced for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.github.io",
        "https://*.netlify.app",
        "https://*.vercel.app",
        "*"  # For development - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HEALTH CHECK ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Physical AI & Humanoid Robotics Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "search": "/search-content",
            "chapters": "/chapters",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Enhanced health check with service status"""
    health_status = {
        "status": "healthy",
        "services": {
            "database": "unknown",
            "qdrant": "unknown",
            "gemini": "unknown"
        },
        "metadata": {
            "vector_count": 0,
            "chapters_indexed": 0
        }
    }
    
    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
        
        # Count chapters
        chapter_count = db.query(models.Chapter).count()
        health_status["metadata"]["chapters_indexed"] = chapter_count
        
        db.close()
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)[:50]}"
        health_status["status"] = "degraded"
    
    # Check Qdrant
    try:
        if vector_db.client:
            vector_count = vector_db.get_vector_count()
            health_status["services"]["qdrant"] = "healthy"
            health_status["metadata"]["vector_count"] = vector_count
        else:
            health_status["services"]["qdrant"] = "disconnected"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["qdrant"] = f"unhealthy: {str(e)[:50]}"
        health_status["status"] = "degraded"
    
    # Check Gemini (simple check)
    try:
        from .gemini_client import gemini_client
        if gemini_client.model:
            health_status["services"]["gemini"] = "healthy"
        else:
            health_status["services"]["gemini"] = "not initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["gemini"] = f"unhealthy: {str(e)[:50]}"
        health_status["status"] = "degraded"
    
    return health_status


# ==================== CHAT ENDPOINT ====================

@app.post("/chat", response_model=schemas.ChatResponse)
async def chat_with_bot(
    chat_request: schemas.ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Enhanced chat endpoint with better error handling and logging
    """
    try:
        # Validate input
        if not chat_request.query or len(chat_request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(chat_request.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        # Log request
        structured_logger.info(
            "chat_endpoint_called",
            user_id=chat_request.user_id or "anonymous",
            query_length=len(chat_request.query),
            chapter_id=chat_request.chapter_id
        )
        
        # Get response from RAG service
        response = rag_service.get_response(
            query=chat_request.query,
            user_id=chat_request.user_id,
            chapter_id=chat_request.chapter_id,
            threshold=0.70,  # Quality threshold
            max_results=5
        )
        
        # Log response
        structured_logger.info(
            "chat_response_sent",
            user_id=chat_request.user_id or "anonymous",
            source_type=response.source_type,
            confidence=response.confidence_score,
            query_time_ms=response.query_time_ms,
            references_count=len(response.references)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.log_error(
            error_type="CHAT_ENDPOINT_ERROR",
            error_message=str(e),
            endpoint="/chat",
            user_id=chat_request.user_id
        )
        
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again."
        )


# ==================== SEARCH ENDPOINT ====================

@app.post("/search-content", response_model=schemas.SearchResponse)
async def search_book_content(
    search_request: schemas.SearchRequest,
    db: Session = Depends(get_db)
):
    """Enhanced search endpoint"""
    try:
        # Validate input
        if not search_request.query or len(search_request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        structured_logger.info(
            "search_endpoint_called",
            query=search_request.query[:100],
            max_results=search_request.max_results
        )
        
        # Get search results
        search_results = rag_service.get_search_results(
            query=search_request.query,
            threshold=search_request.threshold,
            max_results=search_request.max_results
        )
        
        return schemas.SearchResponse(
            results=search_results,
            total_count=len(search_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.log_error(
            error_type="SEARCH_ENDPOINT_ERROR",
            error_message=str(e),
            endpoint="/search-content"
        )
        
        raise HTTPException(
            status_code=500,
            detail="An error occurred during search. Please try again."
        )


# ==================== CHAPTER ENDPOINTS ====================

@app.get("/chapters", response_model=schemas.ChaptersListResponse)
async def get_chapters(
    module: Optional[int] = None,
    limit: int = 37,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get list of chapters with optional module filter"""
    try:
        query = db.query(models.Chapter)
        
        if module is not None:
            if module < 1 or module > 4:
                raise HTTPException(status_code=400, detail="Module must be between 1 and 4")
            query = query.filter(models.Chapter.module == module)
        
        total_count = query.count()
        chapters = query.offset(offset).limit(limit).all()
        
        chapter_summaries = [
            schemas.ChapterSummary(
                id=chapter.id,
                title=chapter.title,
                module=chapter.module,
                description=chapter.description[:200] if chapter.description else None
            )
            for chapter in chapters
        ]
        
        return schemas.ChaptersListResponse(
            chapters=chapter_summaries,
            total_count=total_count,
            offset=offset,
            limit=limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chapters: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching chapters"
        )


@app.get("/chapters/{chapter_id}", response_model=schemas.ChapterDetail)
async def get_chapter_detail(chapter_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific chapter"""
    try:
        chapter = db.query(models.Chapter).filter(models.Chapter.id == chapter_id).first()
        
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")
        
        return schemas.ChapterDetail(
            id=chapter.id,
            title=chapter.title,
            module=chapter.module,
            content=chapter.content,
            description=chapter.description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chapter {chapter_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching chapter details"
        )


# ==================== EMBEDDING ENDPOINTS ====================

@app.post("/embed-chapters", response_model=schemas.EmbedResponse)
async def embed_chapters(
    embed_request: schemas.EmbedRequest,
    db: Session = Depends(get_db)
):
    """
    Create embeddings for specified chapters
    NOTE: This is resource-intensive. Use carefully.
    """
    try:
        if not embed_request.chapter_ids or len(embed_request.chapter_ids) == 0:
            raise HTTPException(status_code=400, detail="chapter_ids cannot be empty")
        
        if len(embed_request.chapter_ids) > 50:
            raise HTTPException(
                status_code=400, 
                detail="Cannot process more than 50 chapters at once"
            )
        
        structured_logger.info(
            "embedding_started",
            chapter_count=len(embed_request.chapter_ids),
            force_rebuild=embed_request.force_rebuild
        )
        
        result = embedding_service.create_embeddings_for_chapters(
            chapter_ids=embed_request.chapter_ids,
            force_rebuild=embed_request.force_rebuild
        )
        
        return schemas.EmbedResponse(
            processed_count=result["processed_count"],
            status=result["status"],
            message=f"Processed {result['processed_count']} chapters successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        structured_logger.log_error(
            error_type="EMBEDDING_ERROR",
            error_message=str(e),
            endpoint="/embed-chapters"
        )
        
        raise HTTPException(
            status_code=500,
            detail="Error creating embeddings"
        )


@app.get("/vector-stats")
async def get_vector_stats():
    """Get statistics about vector database"""
    try:
        stats = {
            "total_vectors": vector_db.get_vector_count(),
            "chapters_indexed": len(vector_db.get_all_chapter_ids()),
            "collection_name": vector_db.collection_name,
            "status": "connected" if vector_db.client else "disconnected"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching vector stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching vector statistics"
        )


# ==================== FAVICON ENDPOINT ====================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests to prevent 404 errors"""
    return Response(status_code=204)


# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 70)
    logger.info("STARTING: Physical AI & Humanoid Robotics Chatbot API")
    logger.info("=" * 70)
    
    # Check Qdrant connection
    if vector_db.client:
        vector_count = vector_db.get_vector_count()
        logger.info(f"SUCCESS: Qdrant connected: {vector_count} vectors indexed")
    else:
        logger.warning("WARN: Qdrant not connected - chatbot will use fallback mode")
    
    # Check database
    try:
        db = SessionLocal()
        chapter_count = db.query(models.Chapter).count()
        logger.info(f"SUCCESS: Database connected: {chapter_count} chapters available")
        db.close()
    except Exception as e:
        logger.error(f"ERROR: Database connection failed: {str(e)}")
    
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("🛑 Shutting down API...")


# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler - FIXED VERSION"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": {
                "chat": "/chat",
                "search": "/search-content",
                "chapters": "/chapters",
                "health": "/health"
            }
        }
    )


# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


# uvicorn app.main:app --reload --port 8000

# http://localhost:8000/docs

# python -m app.scripts.index_markdown

# uvicorn app.main:app --reload --port 8000

# curl http://localhost:8000/check-methods

# curl -X POST http://localhost:8000/test-embedding

# curl -X POST http://localhost:8000/test-one-chapter

# curl -X POST http://localhost:8000/index-markdown