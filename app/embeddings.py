# embeddings.py - ENHANCED VERSION with Chunk Overlap & Better Metadata

from typing import List, Dict, Any, Optional
from .vector_db import vector_db
from .gemini_client import gemini_client
from .models import Chapter
from .database import SessionLocal
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.vector_db = vector_db
        self.gemini_client = gemini_client

    def create_chapter_embeddings(self, chapter: Chapter, force_rebuild: bool = False):
        """
        Create embeddings for a chapter with ENHANCED CHUNKING:
        - Chunk overlap to prevent context loss
        - Better metadata extraction
        - Smarter section splitting
        """
        try:
            # If not force rebuild, check if embeddings exist
            if not force_rebuild:
                existing = self.vector_db.search_similar(
                    query_embedding=[0.1] * 768,
                    threshold=0.0,
                    limit=1,
                    filters={"chapter_id": chapter.id}
                )
                if existing:
                    logger.info(f"Embeddings already exist for chapter {chapter.id}, skipping")
                    return {
                        "chapter_id": chapter.id,
                        "sections_processed": len(existing),
                        "status": "skipped",
                        "message": "Embeddings already exist"
                    }

            # 🎯 ENHANCED CHUNKING with overlap
            sections = self._smart_split_content(
                content=chapter.content,
                chunk_size=800,  # Smaller for precision
                chunk_overlap=200  # 25% overlap
            )

            # Extract chapter metadata
            chapter_metadata = self._extract_chapter_metadata(chapter)

            chapter_embeddings = []
            section_ids = []
            contents = []
            metadatas = []

            for i, section_data in enumerate(sections):
                section_id = f"section_{i+1}"

                # Generate embedding for the section
                embedding = self.gemini_client.embed_content(section_data["text"])

                if embedding:
                    chapter_embeddings.append(embedding)
                    section_ids.append(f"{chapter.id}_{section_id}")
                    contents.append(section_data["text"])
                    
                    # 🎯 ENHANCED METADATA
                    section_metadata = {
                        **chapter_metadata,  # Chapter-level metadata
                        "section_id": section_id,
                        "section_title": section_data.get("title", ""),
                        "subsection": section_data.get("subsection", ""),
                        "content_type": section_data.get("type", "general"),
                        "has_code": section_data.get("has_code", False),
                        "word_count": len(section_data["text"].split()),
                        "created_at": datetime.utcnow().isoformat()
                    }
                    metadatas.append(section_metadata)

            # Add all embeddings to vector database
            if chapter_embeddings:
                success_count = self.vector_db.add_batch_embeddings(
                    chapter_ids=[chapter.id] * len(chapter_embeddings),
                    section_ids=section_ids,
                    contents=contents,
                    embeddings=chapter_embeddings,
                    metadatas=metadatas
                )

                logger.info(f"Created {success_count} embeddings for chapter {chapter.id}")
                return {
                    "chapter_id": chapter.id,
                    "sections_processed": success_count,
                    "status": "success"
                }
            else:
                logger.warning(f"No embeddings created for chapter {chapter.id}")
                return {
                    "chapter_id": chapter.id,
                    "sections_processed": 0,
                    "status": "failed",
                    "message": "No embeddings could be generated"
                }

        except Exception as e:
            logger.error(f"Error creating embeddings for chapter {chapter.id}: {str(e)}")
            return {
                "chapter_id": chapter.id,
                "sections_processed": 0,
                "status": "error",
                "message": str(e)
            }

    def _extract_chapter_metadata(self, chapter: Chapter) -> dict:
        """
        Extract rich metadata from chapter
        """
        # Parse chapter number (e.g., "1", "12", "Chapter 5")
        chapter_num_match = re.search(r'(\d+)', chapter.id)
        chapter_number = chapter_num_match.group(1) if chapter_num_match else ""
        
        # Map module numbers to names
        module_names = {
            1: "Module 1: ROS 2 Fundamentals",
            2: "Module 2: Gazebo & Unity Simulation",
            3: "Module 3: NVIDIA Isaac Platform",
            4: "Module 4: Vision-Language-Action Models"
        }
        
        module_name = module_names.get(chapter.module, f"Module {chapter.module}")
        
        # Extract difficulty from content (heuristic)
        difficulty = self._estimate_difficulty(chapter.content)
        
        # Extract prerequisites from title/content
        prerequisites = self._extract_prerequisites(chapter.title, chapter.content)
        
        return {
            "chapter_number": chapter_number,
            "chapter_title": chapter.title,
            "module": module_name,
            "module_number": chapter.module,
            "difficulty": difficulty,
            "prerequisites": prerequisites,
            "word_count": chapter.word_count if hasattr(chapter, 'word_count') else len(chapter.content.split())
        }

    def _smart_split_content(
        self,
        content: str,
        chunk_size: int = 800,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Smart content splitting with:
        - Overlap to maintain context
        - Respect section boundaries
        - Preserve code blocks
        - Extract metadata per chunk
        """
        sections = []
        
        # Step 1: Split by major sections (headers)
        major_sections = self._split_by_headers(content)
        
        for section_dict in major_sections:
            section_text = section_dict["text"]
            section_title = section_dict["title"]
            
            # If section is small enough, keep it as-is
            if len(section_text) <= chunk_size:
                sections.append({
                    "text": section_text,
                    "title": section_title,
                    "type": self._detect_content_type(section_text),
                    "has_code": "```" in section_text or "def " in section_text
                })
                continue
            
            # Step 2: Split large sections with overlap
            chunks = self._chunk_with_overlap(section_text, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                sections.append({
                    "text": chunk,
                    "title": section_title,
                    "subsection": f"Part {i+1}" if len(chunks) > 1 else "",
                    "type": self._detect_content_type(chunk),
                    "has_code": "```" in chunk or "def " in chunk
                })
        
        return sections

    def _split_by_headers(self, content: str) -> List[Dict[str, str]]:
        """Split content by markdown headers"""
        sections = []
        current_section = {"title": "Introduction", "text": ""}
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if line is a header (##, ###, ####)
            header_match = re.match(r'^(#{2,4})\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_section["text"].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": header_match.group(2).strip(),
                    "text": ""
                }
            else:
                current_section["text"] += line + "\n"
        
        # Add last section
        if current_section["text"].strip():
            sections.append(current_section)
        
        # If no headers found, return whole content
        if not sections:
            sections = [{"title": "Content", "text": content}]
        
        return sections

    def _chunk_with_overlap(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Chunk text with overlap, respecting sentence boundaries
        """
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # If adding this sentence exceeds chunk_size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # 🎯 OVERLAP: Go back by overlap amount
                overlap_text = ""
                j = i - 1
                while j >= 0 and len(overlap_text) < overlap:
                    overlap_text = sentences[j] + " " + overlap_text
                    j -= 1
                
                current_chunk = overlap_text + sentence + " "
            else:
                current_chunk += sentence + " "
            
            i += 1
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _detect_content_type(self, text: str) -> str:
        """Detect type of content"""
        text_lower = text.lower()
        
        if "```" in text or "def " in text or "class " in text:
            return "code_example"
        elif "introduction" in text_lower or "overview" in text_lower:
            return "introduction"
        elif "exercise" in text_lower or "practice" in text_lower:
            return "exercise"
        elif "example" in text_lower:
            return "example"
        elif "tutorial" in text_lower or "step" in text_lower:
            return "tutorial"
        else:
            return "general"

    def _estimate_difficulty(self, content: str) -> str:
        """Estimate difficulty level based on content"""
        content_lower = content.lower()
        
        # Check for advanced keywords
        advanced_keywords = ["advanced", "optimization", "performance", "complex", "distributed"]
        beginner_keywords = ["introduction", "basics", "getting started", "fundamentals"]
        
        advanced_count = sum(1 for kw in advanced_keywords if kw in content_lower)
        beginner_count = sum(1 for kw in beginner_keywords if kw in content_lower)
        
        if advanced_count > beginner_count:
            return "advanced"
        elif beginner_count > 0:
            return "beginner"
        else:
            return "intermediate"

    def _extract_prerequisites(self, title: str, content: str) -> List[str]:
        """Extract prerequisites from content"""
        prereqs = []
        
        # Look for "prerequisite" or "requires" patterns
        prereq_pattern = r'(?:prerequisite|requires|you should know|familiar with)[:\s]+([^.]+)'
        matches = re.findall(prereq_pattern, content.lower())
        
        for match in matches[:3]:  # Max 3 prereqs
            prereqs.append(match.strip())
        
        return prereqs

    def create_embeddings_for_chapters(self, chapter_ids: List[str], force_rebuild: bool = False) -> Dict[str, Any]:
        """Create embeddings for multiple chapters"""
        db: Session = SessionLocal()
        results = []
        processed_count = 0
        failed_count = 0

        try:
            for chapter_id in chapter_ids:
                chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()

                if not chapter:
                    results.append({
                        "chapter_id": chapter_id,
                        "status": "error",
                        "message": "Chapter not found"
                    })
                    failed_count += 1
                    continue

                result = self.create_chapter_embeddings(chapter, force_rebuild)
                results.append(result)

                if result["status"] == "success":
                    processed_count += 1
                else:
                    failed_count += 1

            status = "success" if failed_count == 0 else ("partial" if processed_count > 0 else "error")

            return {
                "processed_count": processed_count,
                "failed_count": failed_count,
                "total_requested": len(chapter_ids),
                "status": status,
                "details": results
            }

        finally:
            db.close()

    def search_similar_content(self, query: str, threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for content similar to the query"""
        try:
            query_embedding = self.gemini_client.embed_content(query)

            if not query_embedding:
                logger.warning("Could not generate embedding for query")
                return []

            results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=limit
            )

            return results

        except Exception as e:
            logger.error(f"Error searching for similar content: {str(e)}")
            return []

    def embed_text(self, text: str):
        """Generate embedding for a text string"""
        try:
            embedding = self.gemini_client.embed_content(text)
            if embedding is None or (isinstance(embedding, list) and len(embedding) == 0):
                return None
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            return None


# Global instance
embedding_service = EmbeddingService()