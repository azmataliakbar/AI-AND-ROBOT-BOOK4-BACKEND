# main.py

import warnings

# Set warning filters as early as possible to catch import-time warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*google.generativeai.*")

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from pathlib import Path
from dotenv import load_dotenv
import logging
import traceback
import glob
import os

# Local imports
from . import models, schemas
from .database import engine, Base, get_db
from .embeddings import embedding_service
from .progress_service import progress_service
from .rag import rag_service
from .scripts.index_markdown import parse_info, chunk_text  # helpers for indexing

# ============================================================
# Environment & Logging
# ============================================================
load_dotenv()

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("⚠️ GEMINI_API_KEY not found in environment variables")
else:
    print("✅ GEMINI_API_KEY loaded")

# ============================================================
# 🚀 STARTUP BANNER
# ============================================================
print("\n" + "=" * 70)
print("🤖  AI & ROBOTICS BOOK PLATFORM API")
print("=" * 70)
print("📚  Physical AI & Humanoid Robotics Educational Platform")
print("🔧  FastAPI + Gemini AI + Qdrant + PostgreSQL")
print("=" * 70 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified!")
    print("✅ Application security ready!")
    print("\n" + "=" * 70)
    print("🎉  ALL SERVICES INITIALIZED SUCCESSFULLY!")
    print("=" * 70)
    print("📡  API Documentation: http://localhost:8000/docs")
    print("📊  Health Check: http://localhost:8000/")
    print("💬  Chat Endpoint: http://localhost:8000/chat")
    print("=" * 70 + "\n")
    yield
    print("\n" + "=" * 70)
    print("👋  Shutting down AI & Robotics Book Platform API")
    print("=" * 70 + "\n")


app = FastAPI(
    title="AI and Robotics Book Platform API",
    description="API for the interactive AI-powered book platform featuring robotics education",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "https://yourdomain.com"],  # Restrict to known origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================
# Error Handlers
# ============================================================
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url.path}: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code} at {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error at {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )

# ============================================================
# Root Endpoint
# ============================================================
@app.get("/")
async def root():
    return {
        "message": "AI and Robotics Book Platform API",
        "status": "online",
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "vector_db": "connected",
            "ai_model": "gemini-2.5-flash",
        },
    }

# ============================================================
# Chat Endpoint
# ============================================================
@app.post("/chat", response_model=schemas.ChatResponse)
async def chat_with_book(chat_request: schemas.ChatRequest, db: Session = Depends(get_db)):
    try:
        response = rag_service.get_response(
            query=chat_request.query,
            user_id=chat_request.user_id,
            chapter_id=chat_request.chapter_id,
        )
        return response
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"/chat error: {exc}")
        logger.error(traceback.format_exc())
        # Return a known-safe shape matching ChatResponse
        return schemas.ChatResponse(
            response="I encountered an error while processing your request. Please try again.",
            confidence_score=0.0,
            source_type="fallback",
            references=[],
            query_time_ms=0,
        )

# ============================================================
# Search Endpoint
# ============================================================
@app.post("/search-content", response_model=schemas.SearchResponse)
async def search_book_content(search_request: schemas.SearchRequest, db: Session = Depends(get_db)):
    try:
        search_results = rag_service.get_search_results(
            query=search_request.query,
            threshold=search_request.threshold,
            max_results=search_request.max_results,
        )
        return schemas.SearchResponse(
            results=search_results,
            total_count=len(search_results),
        )
    except Exception as exc:
        logger.error(f"/search-content error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Search failed")

# ============================================================
# Embedding Endpoint
# ============================================================
@app.post("/embed-chapters", response_model=schemas.EmbedResponse)
async def embed_chapters(embed_request: schemas.EmbedRequest, db: Session = Depends(get_db)):
    try:
        result = embedding_service.create_embeddings_for_chapters(
            chapter_ids=embed_request.chapter_ids,
            force_rebuild=embed_request.force_rebuild,
        )
        return schemas.EmbedResponse(
            processed_count=result["processed_count"],
            status=result["status"],
            message=f"Processed {result['processed_count']} chapters successfully",
        )
    except Exception as exc:
        logger.error(f"/embed-chapters error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Embedding process failed")

# ============================================================
# Progress Tracking Endpoints
# ============================================================
@app.post("/progress", response_model=schemas.ProgressTrackerResponse)
async def track_progress(progress_data: schemas.ProgressTrackerCreate, db: Session = Depends(get_db)):
    try:
        result = progress_service.update_progress(progress_data)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update progress")
        return result
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"/progress error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to update progress")


@app.get("/progress/{user_id}/{chapter_id}", response_model=schemas.ProgressTrackerResponse)
async def get_progress(user_id: str, chapter_id: str, db: Session = Depends(get_db)):
    try:
        result = progress_service.get_progress(user_id, chapter_id)
        if not result:
            raise HTTPException(status_code=404, detail="Progress not found")
        return result
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"/progress/{user_id}/{chapter_id} error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to fetch progress")


@app.get("/progress/{user_id}", response_model=dict)
async def get_user_progress_summary(user_id: str, db: Session = Depends(get_db)):
    try:
        result = progress_service.get_user_progress_summary(user_id)
        return result
    except Exception as exc:
        logger.error(f"/progress/{user_id} summary error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to fetch user progress summary")

# ============================================================
# 📚 Index Markdown Endpoint
# ============================================================
FRONTEND_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend" / "docs"

@app.post("/index-markdown")
async def index_markdown_files():
    """Index all markdown chapters into Qdrant"""
    try:
        md_paths = glob.glob(str(FRONTEND_DATA_ROOT / "**" / "chapter*.md"), recursive=True)
        print(f"📂 Looking in: {FRONTEND_DATA_ROOT.resolve()}")
        print(f"🔍 Found {len(md_paths)} markdown files")

        if not md_paths:
            return {"status": "error", "message": "No markdown files found"}

        chapter_ids, section_ids, contents = [], [], []

        for md in md_paths:
            print(f"📄 Processing file: {md}")
            module_id, chapter_id, chapter_title = parse_info(Path(md))
            with open(md, "r", encoding="utf-8") as f:
                text = f.read()

            if len(text) > 10_000_000:
                print(f"⚠️ Skipping {md} (too large: {len(text)} chars)")
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chapter_ids.append(chapter_id)
                section_ids.append(f"{module_id}-{chapter_id}-{i}")
                contents.append(chunk)

            print(f"   ✅ Generated {len(chunks)} chunks for {md}")

        if not contents:
            return {"status": "error", "message": "No content chunks generated"}

        print(f"🚀 Embedding {len(contents)} chunks into Qdrant...")
        embedding_service.create_embeddings_for_texts(chapter_ids, section_ids, contents)
        print(f"✅ Finished indexing {len(contents)} chunks into Qdrant")

        return {"status": "success", "message": f"Indexed {len(contents)} chunks into Qdrant"}
    except Exception as exc:
        logger.error(f"/index-markdown error: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Indexing markdown failed")

# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn app.main:app --reload --port 8000

# http://localhost:8000/docs

# python -m app.scripts.index_markdown

# uvicorn app.main:app --reload --port 8000

# curl http://localhost:8000/check-methods

# curl -X POST http://localhost:8000/test-embedding

# curl -X POST http://localhost:8000/test-one-chapter

# curl -X POST http://localhost:8000/index-markdown