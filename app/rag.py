# rag.py - ENHANCED VERSION with Query Classification & Smart Filtering

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
from time import perf_counter
import re

from .gemini_client import gemini_client
from .models import Chapter
from .database import SessionLocal
from .schemas import ChatResponse, Reference, SearchResult
from .logging_config import logger as structured_logger
from .vector_db import vector_db
from .embeddings import embedding_service

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classify queries to apply intelligent filtering"""
    
    @staticmethod
    def classify(query: str) -> Dict[str, Any]:
        """
        Classify query and extract filters
        
        Returns:
        {
            "type": "overview|specific_chapter|specific_module|concept|general",
            "filters": {...},
            "intent": "..."
        }
        """
        query_lower = query.lower()
        
        classification = {
            "type": "general",
            "filters": {},
            "intent": "answer_question",
            "needs_summary": False
        }
        
        # Pattern 1: Module-specific queries
        module_patterns = [
            r"module\s+(\d+)",
            r"module\s+(one|two|three|four|1|2|3|4)",
            r"(first|second|third|fourth)\s+module"
        ]
        
        for pattern in module_patterns:
            match = re.search(pattern, query_lower)
            if match:
                module_num = QueryClassifier._parse_module_number(match.group(1))
                if module_num:
                    classification["type"] = "specific_module"
                    classification["filters"]["module_number"] = module_num
                    classification["needs_summary"] = "tell me about" in query_lower or "overview" in query_lower
                    break
        
        # Pattern 2: Chapter-specific queries
        chapter_patterns = [
            r"chapter\s+(\d+)",
            r"ch\s+(\d+)",
            r"chapter\s+(one|two|three|[\w]+)"
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    classification["type"] = "specific_chapter"
                    classification["filters"]["chapter_number"] = str(chapter_num)
                    break
                except ValueError:
                    pass
        
        # Pattern 3: Overview/counting queries
        overview_patterns = [
            r"how many (modules|chapters)",
            r"list (all )?(modules|chapters)",
            r"what (modules|chapters)",
            r"overview of",
            r"structure of",
            r"book organization"
        ]
        
        for pattern in overview_patterns:
            if re.search(pattern, query_lower):
                classification["type"] = "overview"
                classification["needs_summary"] = True
                classification["intent"] = "list_structure"
                break
        
        # Pattern 4: Concept queries (what is, explain, define)
        concept_patterns = [
            r"what is",
            r"explain",
            r"define",
            r"tell me about"
        ]
        
        for pattern in concept_patterns:
            if re.search(pattern, query_lower):
                if classification["type"] == "general":
                    classification["type"] = "concept"
                break
        
        # Pattern 5: Extract topics for filtering
        topics = QueryClassifier._extract_topics(query)
        if topics:
            classification["filters"]["topics"] = topics
        
        return classification
    
    @staticmethod
    def _parse_module_number(module_str: str) -> Optional[int]:
        """Convert module string to number"""
        module_map = {
            "one": 1, "1": 1,
            "two": 2, "2": 2,
            "three": 3, "3": 3,
            "four": 4, "4": 4,
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4
        }
        return module_map.get(module_str.lower())
    
    @staticmethod
    def _extract_topics(query: str) -> List[str]:
        """Extract key robotics topics from query"""
        topic_keywords = {
            "ROS2": ["ros2", "ros 2", "robot operating system"],
            "Gazebo": ["gazebo", "simulation"],
            "Unity": ["unity"],
            "NVIDIA Isaac": ["isaac", "nvidia"],
            "nodes": ["nodes", "node"],
            "topics": ["topics", "topic", "publisher", "subscriber"],
            "services": ["services", "service"],
            "actions": ["actions", "action"],
            "VLA": ["vla", "vision-language-action"],
            "physics": ["physics", "collision", "gravity"],
            "URDF": ["urdf", "robot model"]
        }
        
        query_lower = query.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_topics.append(topic)
                    break
        
        return found_topics[:3]  # Max 3 topics


class RAGService:
    def __init__(self):
        self.embedding_service = embedding_service
        self.gemini_client = gemini_client
        self.vector_db = vector_db
        self.query_classifier = QueryClassifier()

    def get_response(
        self,
        query: str,
        user_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        threshold: float = 0.70,
        max_results: int = 5
    ) -> ChatResponse:
        """
        Enhanced RAG with INTELLIGENT QUERY CLASSIFICATION:
        1) Classify query type and extract filters
        2) Apply smart vector search with filters
        3) Generate context-aware response
        4) Handle quota gracefully
        """
        start_time = perf_counter()

        try:
            structured_logger.info(
                "chat_request_received",
                user_id=user_id,
                query=query[:100] + "..." if len(query) > 100 else query,
                execution_time=None
            )

            # 🎯 STEP 1: CLASSIFY QUERY
            classification = self.query_classifier.classify(query)
            logger.info(f"Query classified as: {classification['type']}, filters: {classification['filters']}")

            # 🎯 STEP 2: Handle overview queries specially
            if classification["type"] == "overview":
                return self._handle_overview_query(query, user_id, chapter_id, start_time)

            # 🎯 STEP 3: Generate embedding
            query_embedding = self.embedding_service.embed_text(query)

            if query_embedding is None or (isinstance(query_embedding, list) and len(query_embedding) == 0):
                logger.warning(f"Embedding generation failed for user {user_id}, using Gemini fallback")
                return self._handle_fallback_response(query, user_id, chapter_id, start_time)

            # 🎯 STEP 4: SMART VECTOR SEARCH with FILTERS
            vector_results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=max_results * 2 if classification["filters"] else max_results,  # Get more for filtering
                filters=classification["filters"] if classification["filters"] else None
            )

            # If no results with filters, try without filters
            if not vector_results and classification["filters"]:
                logger.info("No results with filters, retrying without filters")
                vector_results = self.vector_db.search_similar(
                    query_embedding=query_embedding,
                    threshold=threshold * 0.9,  # Lower threshold
                    limit=max_results,
                    filters=None
                )

            if vector_results:
                # Build enhanced context
                context = self._build_enhanced_context(vector_results, classification)
                
                # Build query-aware prompt
                enhanced_prompt = self._build_query_aware_prompt(
                    query, 
                    context, 
                    vector_results,
                    classification
                )

                # Try Gemini with enhanced quota handling
                gemini_response = self._safe_gemini(
                    self.gemini_client.chat_with_context,
                    message=enhanced_prompt,
                    context=context
                )

                # ✅ CHECK FOR QUOTA EXCEEDED
                if gemini_response.get("quota_exceeded"):
                    fallback_response = self._build_quota_fallback_response(
                        query=query,
                        context=context,
                        vector_results=vector_results,
                        classification=classification
                    )
                    
                    references = self._build_enhanced_references(vector_results)
                    query_time_ms = int((perf_counter() - start_time) * 1000)

                    structured_logger.log_chat_interaction(
                        user_id=user_id or "anonymous",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        response=fallback_response[:200],
                        confidence_score=0.5,
                        execution_time=query_time_ms,
                        source_type="quota_fallback",
                        chapter_id=chapter_id
                    )

                    return ChatResponse(
                        response=fallback_response,
                        confidence_score=0.5,
                        source_type="quota_fallback",
                        references=references,
                        query_time_ms=query_time_ms
                    )

                # Build rich references with citations
                references = self._build_enhanced_references(vector_results)

                # Add citation markers to response
                enhanced_response = self._add_citations_to_response(
                    gemini_response["response"],
                    vector_results
                )

                query_time_ms = int((perf_counter() - start_time) * 1000)

                structured_logger.log_chat_interaction(
                    user_id=user_id or "anonymous",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    response=enhanced_response[:200],
                    confidence_score=gemini_response["confidence"],
                    execution_time=query_time_ms,
                    source_type="qdrant",
                    chapter_id=chapter_id
                )

                return ChatResponse(
                    response=enhanced_response,
                    confidence_score=gemini_response["confidence"],
                    source_type="qdrant",
                    references=references,
                    query_time_ms=query_time_ms
                )
            else:
                # No vector results → Gemini fallback
                return self._handle_fallback_response(query, user_id, chapter_id, start_time)

        except Exception as e:
            query_time_ms = int((perf_counter() - start_time) * 1000)

            structured_logger.log_error(
                error_type="RAG_SERVICE_ERROR",
                error_message=str(e),
                endpoint="/chat",
                user_id=user_id
            )

            return ChatResponse(
                response="I encountered an error while processing your request. Please try again.",
                confidence_score=0.0,
                source_type="error",
                references=[],
                query_time_ms=query_time_ms
            )

    def _handle_overview_query(
        self,
        query: str,
        user_id: Optional[str],
        chapter_id: Optional[str],
        start_time: float
    ) -> ChatResponse:
        """
        Handle overview queries like "how many modules?"
        Returns structured information instead of searching
        """
        db: Session = SessionLocal()
        try:
            # Get database statistics
            total_chapters = db.query(Chapter).count()
            
            # Count by modules
            modules_data = {}
            for module_num in range(1, 5):
                module_chapters = db.query(Chapter).filter(Chapter.module == module_num).all()
                modules_data[module_num] = {
                    "count": len(module_chapters),
                    "titles": [ch.title for ch in module_chapters[:3]]  # First 3
                }
            
            # Build structured response
            response = f"""**📚 Physical AI & Humanoid Robotics Book Structure**

**Total Chapters:** {total_chapters} chapters organized into 4 modules

**Module Breakdown:**

🤖 **Module 1: ROS 2 Fundamentals** - {modules_data[1]['count']} chapters
   • Foundation concepts, nodes, topics, services, and actions
   • First few chapters: {', '.join(modules_data[1]['titles'][:3])}

🎮 **Module 2: Gazebo & Unity Simulation** - {modules_data[2]['count']} chapters
   • Physics simulation, robot modeling, and virtual environments
   • First few chapters: {', '.join(modules_data[2]['titles'][:3])}

🚀 **Module 3: NVIDIA Isaac Platform** - {modules_data[3]['count']} chapters
   • Isaac Sim, Isaac Lab, GPU-accelerated robotics
   • First few chapters: {', '.join(modules_data[3]['titles'][:3])}

👁️ **Module 4: Vision-Language-Action Models** - {modules_data[4]['count']} chapters
   • VLA models, multimodal AI, natural language control
   • First few chapters: {', '.join(modules_data[4]['titles'][:3])}

**Learning Path:** Start with Module 1 and progress sequentially for the best learning experience!"""

            query_time_ms = int((perf_counter() - start_time) * 1000)

            structured_logger.log_chat_interaction(
                user_id=user_id or "anonymous",
                query=query,
                response=response[:200],
                confidence_score=1.0,
                execution_time=query_time_ms,
                source_type="structured_overview",
                chapter_id=chapter_id
            )

            return ChatResponse(
                response=response,
                confidence_score=1.0,
                source_type="structured_overview",
                references=[],
                query_time_ms=query_time_ms
            )
        finally:
            db.close()

    def _handle_fallback_response(
        self,
        query: str,
        user_id: Optional[str],
        chapter_id: Optional[str],
        start_time: float
    ) -> ChatResponse:
        """Handle fallback when no vector results or embedding fails"""
        gemini_response = self._safe_gemini(
            self.gemini_client.generate_content, 
            prompt=query
        )

        if gemini_response.get("quota_exceeded"):
            fallback_message = self._build_out_of_scope_quota_message()
            query_time_ms = int((perf_counter() - start_time) * 1000)

            return ChatResponse(
                response=fallback_message,
                confidence_score=0.0,
                source_type="out_of_scope_quota_exceeded",
                references=[],
                query_time_ms=query_time_ms
            )

        query_time_ms = int((perf_counter() - start_time) * 1000)

        structured_logger.log_chat_interaction(
            user_id=user_id or "anonymous",
            query=query,
            response=gemini_response["response"],
            confidence_score=gemini_response["confidence"],
            execution_time=query_time_ms,
            source_type="fallback",
            chapter_id=chapter_id
        )

        return ChatResponse(
            response=gemini_response["response"],
            confidence_score=gemini_response["confidence"],
            source_type="fallback",
            references=[],
            query_time_ms=query_time_ms
        )

    def _build_enhanced_context(
        self, 
        vector_results: List[dict],
        classification: Dict[str, Any]
    ) -> str:
        """
        Build context with awareness of query type
        """
        context_parts = []
        
        # Group by chapter for better organization
        chapters_seen = set()
        
        for i, result in enumerate(vector_results[:5], 1):  # Top 5 results
            chapter_num = result.get("chapter_number", "")
            chapter_title = result.get("chapter_title", "Unknown Chapter")
            module = result.get("module", "")
            content = result.get("content", "")
            
            # Avoid duplicate chapter headers
            chapter_key = f"{chapter_num}:{chapter_title}"
            if chapter_key not in chapters_seen:
                section_header = f"\n=== Chapter {chapter_num}: {chapter_title}"
                if module:
                    section_header += f" ({module})"
                section_header += " ===\n"
                context_parts.append(section_header)
                chapters_seen.add(chapter_key)
            
            context_parts.append(content + "\n")
        
        return "\n".join(context_parts)

    def _build_query_aware_prompt(
        self,
        query: str,
        context: str,
        vector_results: List[dict],
        classification: Dict[str, Any]
    ) -> str:
        """Build prompt based on query classification"""
        
        # Get chapter info
        chapters_info = []
        for result in vector_results[:3]:
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_num and ch_title:
                chapters_info.append(f"Chapter {ch_num}: {ch_title}")
        
        chapters_str = ", ".join(chapters_info) if chapters_info else "the book"
        
        # Build prompt based on classification
        if classification["type"] == "specific_module":
            module_num = classification["filters"].get("module_number", "")
            prompt = f"""You are an AI assistant helping students learn Physical AI and Humanoid Robotics.

The user asked: "{query}"

They specifically asked about MODULE {module_num}. I found relevant content from {chapters_str}.

Provide a comprehensive answer about MODULE {module_num} ONLY, based on the book content below. Focus on the chapters, topics, and concepts covered in this module.

Book Content:
{context}

Provide a detailed, focused response about Module {module_num}:"""
        
        elif classification["type"] == "specific_chapter":
            chapter_num = classification["filters"].get("chapter_number", "")
            prompt = f"""You are an AI assistant helping students learn Physical AI and Humanoid Robotics.

The user asked: "{query}"

They asked about CHAPTER {chapter_num}. Provide a comprehensive answer based on the content below.

Book Content:
{context}

Provide a detailed response about Chapter {chapter_num}:"""
        
        else:
            # General concept or question
            prompt = f"""You are an AI assistant helping students learn Physical AI and Humanoid Robotics.

The user asked: "{query}"

I found relevant information from {chapters_str}.

Provide a comprehensive, educational answer based ONLY on the book content below. When referencing information, mention which chapter it comes from.

Book Content:
{context}

Provide a detailed response:"""
        
        return prompt

    def _build_quota_fallback_response(
        self, 
        query: str, 
        context: str, 
        vector_results: List[dict],
        classification: Dict[str, Any] = None
    ) -> str:
        """Build helpful fallback when quota exceeded"""
        chapters_info = []
        for result in vector_results[:3]:
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_num and ch_title:
                chapters_info.append(f"Chapter {ch_num}: {ch_title}")
        
        chapters_str = ", ".join(chapters_info) if chapters_info else "relevant chapters"
        
        # Add filter context if present
        filter_context = ""
        if classification and classification.get("filters"):
            if "module_number" in classification["filters"]:
                filter_context = f" (Module {classification['filters']['module_number']})"
        
        content_preview = context[:1000] if len(context) > 1000 else context
        
        fallback_response = f"""**[AI Summary Temporarily Unavailable - Quota Limit Reached]**

Your question: *"{query}"*

I found relevant information from **{chapters_str}**{filter_context}. Here's the content from your textbook:

---

{content_preview}

{'...' if len(context) > 1000 else ''}

---

**Note:** The AI model has reached its daily quota limit. The content above is directly from your Physical AI & Humanoid Robotics textbook. 

**To get a full answer:**
- Wait a few minutes and try again (quota resets periodically)
- Read the complete chapter sections referenced above
- The quota fully resets at midnight PST

📚 **Sources:** {chapters_str}"""
        
        return fallback_response

    def _build_out_of_scope_quota_message(self) -> str:
        """Message when question is out-of-scope AND quota exceeded"""
        return """**📚 Question Not Found in Textbook**

I couldn't find relevant content in the **Physical AI & Humanoid Robotics** textbook for your question.

**This chatbot specializes in:**
- 🤖 **ROS 2 Fundamentals** - Nodes, topics, services, actions
- 🎮 **Gazebo & Unity Simulation** - Physics engines, robot models, environments
- 🚀 **NVIDIA Isaac Platform** - Isaac Sim, Isaac Lab, GPU-accelerated robotics
- 👁️ **Vision-Language-Action Models** - VLA models, multimodal AI, natural language control

**Please ask questions related to the 37 chapters in the textbook.**

*Note: The AI assistant is temporarily unavailable due to quota limits.*"""

    def _build_enhanced_references(self, vector_results: List[dict]) -> List[Reference]:
        """Build rich reference objects"""
        references = []
        
        for result in vector_results:
            ref = Reference(
                chapter_id=result.get("chapter_id"),
                chapter_title=result.get("chapter_title"),
                section=result.get("section_title") or result.get("section_id"),
                chapter_number=result.get("chapter_number"),
                module=result.get("module"),
                similarity_score=result.get("similarity_score"),
                content_preview=result.get("content", "")[:150] + "..."
            )
            references.append(ref)
        
        return references

    def _add_citations_to_response(self, response: str, vector_results: List[dict]) -> str:
        """Add chapter citations"""
        if not vector_results:
            return response
        
        chapters = []
        for result in vector_results[:3]:
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_title:
                if ch_num:
                    chapters.append(f"Chapter {ch_num}: {ch_title}")
                else:
                    chapters.append(ch_title)
        
        if chapters:
            citation_footer = f"\n\n📚 **Sources:** {', '.join(chapters)}"
            return response + citation_footer
        
        return response

    def _safe_gemini(self, fn, **kwargs) -> dict:
        """Wrap Gemini calls for consistent response"""
        try:
            resp = fn(**kwargs)
            if not isinstance(resp, dict):
                raise ValueError("Gemini returned non-dict response")
            
            if resp.get("quota_exceeded"):
                return {
                    "response": resp.get("response", "Quota exceeded"),
                    "confidence": resp.get("confidence", 0.5),
                    "safety_ratings": resp.get("safety_ratings", []),
                    "quota_exceeded": True
                }
            
            return {
                "response": resp.get("response") or resp.get("text") or "",
                "confidence": resp.get("confidence", 0.0),
                "safety_ratings": resp.get("safety_ratings", []),
                "quota_exceeded": False
            }
        except Exception as e:
            error_str = str(e).lower()
            
            if "quota" in error_str or "429" in error_str or "resourceexhausted" in error_str:
                logger.warning(f"Gemini quota exceeded: {str(e)[:100]}")
                return {
                    "response": "[QUOTA_EXCEEDED]",
                    "confidence": 0.0,
                    "safety_ratings": [],
                    "quota_exceeded": True
                }
            
            logger.error(f"Gemini call failed: {e}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "confidence": 0.0,
                "safety_ratings": [],
                "quota_exceeded": False
            }

    def get_search_results(
        self, 
        query: str, 
        threshold: float = 0.7, 
        max_results: int = 5
    ) -> List[SearchResult]:
        """Get enhanced search results"""
        try:
            query_embedding = self.embedding_service.embed_text(query)

            if query_embedding is None:
                return []

            vector_results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=max_results,
                filters=None
            )

            return [
                SearchResult(
                    chapter_id=result.get("chapter_id"),
                    chapter_number=result.get("chapter_number", ""),
                    chapter_title=result.get("chapter_title"),
                    module=result.get("module", ""),
                    content_snippet=(
                        (result.get("content", "")[:200] + "...") 
                        if len(result.get("content", "")) > 200
                        else result.get("content", "")
                    ),
                    similarity_score=result.get("similarity_score"),
                    topics=result.get("topics", [])
                )
                for result in vector_results
            ]
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []


# Global instance
rag_service = RAGService()