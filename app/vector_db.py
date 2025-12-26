# vector_db.py - ENHANCED VERSION with Rich Metadata

from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import json
import time
import uuid
import re


class VectorDBSettings(BaseSettings):
    qdrant_url: str = "https://your-cluster-url.qdrant.io"
    qdrant_api_key: str = "your_qdrant_api_key"
    qdrant_collection_name: str = "robotics_book"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False
    )


vector_settings = VectorDBSettings()


class VectorDB:
    def __init__(self):
        self.client = None
        self.collection_name = None
        
        try:
            print("CONNECTING: Connecting to Qdrant...")
            print(f"   URL: {vector_settings.qdrant_url[:50]}...")
            print(f"   Collection: {vector_settings.qdrant_collection_name}")
            
            self.client = QdrantClient(
                url=vector_settings.qdrant_url,
                api_key=vector_settings.qdrant_api_key,
                timeout=30,
                check_compatibility=False
            )
            self.collection_name = vector_settings.qdrant_collection_name
            
            # Test connection
            self.client.get_collections()
            print("SUCCESS: Qdrant connected successfully!")
            
            self._ensure_collection_exists()
            
        except Exception as e:
            print(f"ERROR: Qdrant connection failed: {str(e)[:200]}")
            print("   Please check:")
            print("   1. QDRANT_URL in .env file")
            print("   2. QDRANT_API_KEY in .env file")
            print("   3. Internet connection")
            self.client = None
            self.collection_name = None

    def _ensure_collection_exists(self):
        """Verify collection exists and create if needed"""
        if not self.client:
            return

        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"SUCCESS: Collection '{self.collection_name}' ready ({collection_info.points_count} vectors)")
        except Exception as e:
            print(f"WARN: Collection '{self.collection_name}' not found: {str(e)[:100]}")
            print(f"   Creating collection '{self.collection_name}'...")

            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"SUCCESS: Collection '{self.collection_name}' created successfully!")
            except Exception as create_error:
                print(f"ERROR: Failed to create collection: {str(create_error)[:100]}")

    def _create_simple_id(self) -> int:
        """Create a simple numeric ID for Qdrant points"""
        return int(time.time() * 1000000) % 10000000000

    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        # Common robotics terms
        topics = []
        keywords = [
            "ROS2", "ROS 2", "Gazebo", "Unity", "NVIDIA Isaac", "Isaac Sim",
            "DDS", "nodes", "topics", "services", "actions", "URDF", "SDF",
            "simulation", "sensor", "actuator", "controller", "navigation",
            "SLAM", "planning", "perception", "manipulation", "VLA", "vision",
            "language", "transformer", "neural network", "training", "deployment"
        ]
        
        content_lower = content.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                topics.append(keyword)
        
        return list(set(topics))[:10]  # Max 10 topics

    def _make_json_safe(self, data: dict) -> dict:
        """Convert all values in dict to JSON-serializable types"""
        safe_dict = {}
        for key, value in data.items():
            try:
                if value is None:
                    safe_dict[key] = None
                elif isinstance(value, (str, int, float, bool)):
                    safe_dict[key] = value
                elif isinstance(value, (list, dict)):
                    json.dumps(value)
                    safe_dict[key] = value
                else:
                    safe_dict[key] = str(value)[:200]
            except (TypeError, ValueError):
                safe_dict[key] = str(value)[:200] if value is not None else None
        return safe_dict

    def add_embeddings(
        self, 
        chapter_id: str, 
        section_id: str, 
        content: str, 
        embedding: List[float], 
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Add a single embedding with ENHANCED METADATA
        
        Enhanced metadata includes:
        - chapter_number: e.g., "1.1", "2.3"
        - chapter_title: Full chapter name
        - module: Which of the 4 modules
        - section: Subsection name
        - topics: Extracted keywords
        - content_type: "introduction", "tutorial", "example", "exercise"
        - word_count: Length of content
        """
        if not self.client:
            print("WARN: Qdrant not available, skipping embedding")
            return False
            
        if metadata is None:
            metadata = {}

        try:
            # Enhanced metadata structure
            enhanced_metadata = {
                # Core identifiers
                "chapter_id": str(chapter_id)[:50],
                "section_id": str(section_id)[:100],
                "content": str(content)[:1000],  # Increased from 500
                
                # NEW: Chapter structure
                "chapter_number": metadata.get("chapter_number", ""),
                "chapter_title": metadata.get("chapter_title", "")[:200],
                "module": metadata.get("module", "")[:100],
                "module_number": metadata.get("module_number", 0),
                
                # NEW: Section details
                "section_title": metadata.get("section_title", "")[:200],
                "subsection": metadata.get("subsection", "")[:200],
                
                # NEW: Content analysis
                "topics": self._extract_topics(content),
                "content_type": metadata.get("content_type", "general"),
                "word_count": len(content.split()),
                "has_code": "```" in content or "def " in content or "class " in content,
                
                # NEW: Learning context
                "difficulty": metadata.get("difficulty", "intermediate"),
                "prerequisites": metadata.get("prerequisites", []),
                
                # Timestamp
                "indexed_at": time.time()
            }
            
            # Merge with any additional metadata
            safe_metadata = self._make_json_safe(enhanced_metadata)
            if metadata:
                safe_metadata.update(self._make_json_safe(metadata))

            point_id = self._create_simple_id()
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=safe_metadata
                    )
                ]
            )
            return True
            
        except Exception as e:
            print(f"WARN: Error adding embedding: {str(e)[:150]}")
            return False

    def add_batch_embeddings(
        self, 
        chapter_ids: List[str], 
        section_ids: List[str], 
        contents: List[str], 
        embeddings: List[List[float]], 
        metadatas: Optional[List[dict]] = None
    ) -> int:
        """Add multiple embeddings with enhanced metadata"""
        if not self.client:
            print("WARN: Qdrant not available, skipping batch embeddings")
            return 0
            
        if metadatas is None:
            metadatas = [{} for _ in range(len(chapter_ids))]

        successful = 0
        failed = 0
        
        for i, (chapter_id, section_id, content, embedding, metadata) in enumerate(
            zip(chapter_ids, section_ids, contents, embeddings, metadatas)
        ):
            try:
                success = self.add_embeddings(chapter_id, section_id, content, embedding, metadata)
                if success:
                    successful += 1
                    if successful % 10 == 0:
                        print(f"   PROGRESS: Added {successful} embeddings...")
                else:
                    failed += 1
                    
                if i % 20 == 0:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"   ERROR: Chunk {i} failed: {str(e)[:80]}")
                failed += 1
        
        if successful > 0:
            print(f"SUCCESS: Successfully added {successful} embeddings to Qdrant")
        if failed > 0:
            print(f"WARN: Failed to add {failed} embeddings")
            
        return successful

    def search_similar(
        self, 
        query_embedding: List[float], 
        threshold: float = 0.7, 
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[dict]:
        """
        ENHANCED SEARCH with optional filters
        
        filters example:
        {
            "module": "Module 1: ROS 2 Fundamentals",
            "chapter_number": "1.1",
            "content_type": "tutorial",
            "topics": ["ROS2", "nodes"]
        }
        """
        if not self.client:
            print("WARN: Qdrant not available")
            return []

        try:
            # Build Qdrant filter conditions
            query_filter = None
            if filters:
                must_conditions = []
                
                # Exact match filters
                for key in ["module", "chapter_number", "chapter_id", "content_type"]:
                    if key in filters and filters[key]:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=filters[key])
                            )
                        )
                
                # Topic filter (any match)
                if "topics" in filters and filters["topics"]:
                    for topic in filters["topics"]:
                        must_conditions.append(
                            models.FieldCondition(
                                key="topics",
                                match=models.MatchAny(any=[topic])
                            )
                        )
                
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)

            # Execute search
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=threshold,
                query_filter=query_filter,
                with_payload=True
            ).points

            # Return enhanced results
            return [
                {
                    "chapter_id": result.payload.get("chapter_id", ""),
                    "chapter_number": result.payload.get("chapter_number", ""),
                    "chapter_title": result.payload.get("chapter_title", ""),
                    "module": result.payload.get("module", ""),
                    "section_id": result.payload.get("section_id", ""),
                    "section_title": result.payload.get("section_title", ""),
                    "content": result.payload.get("content", ""),
                    "topics": result.payload.get("topics", []),
                    "content_type": result.payload.get("content_type", "general"),
                    "similarity_score": result.score,
                    "word_count": result.payload.get("word_count", 0)
                }
                for result in results
            ]
            
        except Exception as e:
            print(f"WARN: Search error: {str(e)[:150]}")
            return []

    def delete_by_chapter_id(self, chapter_id: str) -> int:
        """Delete all embeddings for a chapter"""
        if not self.client:
            return 0
            
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chapter_id",
                            match=models.MatchValue(value=str(chapter_id))
                        )
                    ]
                ),
                limit=1000
            )

            if isinstance(results, tuple):
                points = results[0]
            else:
                points = results

            point_ids = [point.id for point in points]
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                print(f"SUCCESS: Deleted {len(point_ids)} embeddings for chapter {chapter_id}")
                return len(point_ids)
            return 0
            
        except Exception as e:
            print(f"WARN: Delete error: {str(e)[:100]}")
            return 0

    def get_all_chapter_ids(self) -> List[str]:
        """Get all unique chapter IDs"""
        if not self.client:
            return []
            
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )

            if isinstance(results, tuple):
                points = results[0]
            else:
                points = results

            chapter_ids = set()
            for point in points:
                chapter_id = point.payload.get("chapter_id")
                if chapter_id:
                    chapter_ids.add(str(chapter_id))

            return list(chapter_ids)
            
        except Exception as e:
            print(f"WARN: Get chapters error: {str(e)[:100]}")
            return []

    def get_vector_count(self) -> int:
        """Get total number of vectors in collection"""
        if not self.client:
            return 0
            
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except:
            return 0

    def test_connection(self) -> bool:
        """Test if Qdrant connection works"""
        if not self.client:
            return False
            
        try:
            self.client.get_collections()
            
            test_id = self._create_simple_id()
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=test_id,
                        vector=[0.1] * 768,
                        payload={"test": "connection_test", "timestamp": time.time()}
                    )
                ],
                wait=False
            )
            return True
            
        except Exception as e:
            print(f"ERROR: Connection test failed: {str(e)[:150]}")
            return False


# Global instance
vector_db = VectorDB()

if __name__ == "__main__":
    print("\nTESTING: Testing VectorDB connection...")
    if vector_db.client:
        print(f"SUCCESS: VectorDB ready!")
        print(f"   Collection: {vector_db.collection_name}")
        print(f"   Vectors: {vector_db.get_vector_count()}")
    else:
        print("ERROR: VectorDB not connected")