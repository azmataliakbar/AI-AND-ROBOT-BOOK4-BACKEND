# gemini_client.py - PRODUCTION-READY VERSION
# Optimized for Physical AI & Humanoid Robotics Educational Platform

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import logging
import traceback
import time
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeminiSettings(BaseSettings):
    """Settings for Gemini API"""
    gemini_api_key: str

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=False
    )


# Load settings
gemini_settings = GeminiSettings()


def retry_on_error(max_retries=2, delay=1):
    """
    Decorator to retry Gemini API calls on transient failures
    Does NOT retry on quota errors (fail fast)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Don't retry on quota errors - fail immediately
                    if any(keyword in error_str for keyword in ["quota", "429", "resourceexhausted"]):
                        logger.warning("Quota exceeded - failing fast without retry")
                        raise
                    
                    # Don't retry on safety blocks
                    if "safety" in error_str or "blocked" in error_str:
                        logger.warning("Content blocked by safety filters")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


class GeminiClient:
    """
    Production-ready Gemini client for educational RAG chatbot
    Features:
    - Quota detection and graceful fallback
    - Optimized for educational content
    - Enhanced error handling
    - Confidence scoring
    """
    
    def __init__(self):
        try:
            # Configure Gemini with API key
            genai.configure(api_key=gemini_settings.gemini_api_key)
            
            # ✅ CRITICAL: Use gemini-2.5-flash for higher quotas
            self.model_name = "models/gemini-2.5-flash"
            
            # Generation config - optimized for educational Q&A
            self.generation_config = {
                "temperature": 0.2,           # Lower = more focused, deterministic
                "max_output_tokens": 1500,    # Balanced for detailed answers
                "top_p": 0.85,                # Nucleus sampling
                "top_k": 40                   # Top-k sampling
            }
            
            # Safety settings - balanced for educational content
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Initialize main model
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Initialize chat model (same config)
            self.chat_model = genai.GenerativeModel(
                self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            logger.info(f"SUCCESS: Gemini AI ({self.model_name}) initialized successfully!")
            print(f"SUCCESS: Gemini AI (2.5-flash) ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            logger.error(traceback.format_exc())
            print("WARN: Gemini AI initialization failed")
            raise

    @retry_on_error(max_retries=2, delay=1)
    def generate_content(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate content with retry logic and quota detection
        Returns: Dict with response, confidence, and error flags
        """
        try:
            full_prompt = self._build_prompt(prompt, context, system_instruction)
            
            response = self.model.generate_content(full_prompt)
            
            # Check if response was blocked by safety filters
            if not response.text:
                logger.warning("Response blocked by safety filters")
                return {
                    "text": "I apologize, but I cannot provide a response to this query due to content safety restrictions. Please try rephrasing your question.",
                    "confidence": 0.0,
                    "safety_ratings": self._extract_safety_ratings(response),
                    "blocked": True
                }
            
            response_text = response.text.strip()
            confidence = self._calculate_confidence(response_text, has_context=bool(context))
            
            return {
                "text": response_text,
                "confidence": confidence,
                "safety_ratings": self._extract_safety_ratings(response),
                "blocked": False
            }
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error generating content: {str(e)}")
            
            # ✅ CRITICAL: Detect quota exceeded
            if any(keyword in error_str for keyword in ["quota", "429", "resourceexhausted"]):
                logger.warning("WARN: QUOTA EXCEEDED in generate_content")
                return {
                    "text": "[QUOTA_EXCEEDED]",
                    "confidence": 0.0,
                    "safety_ratings": [],
                    "quota_exceeded": True,
                    "error": str(e)[:300]
                }
            
            # Generic error
            return {
                "text": "I encountered an error while processing your request. Please try again.",
                "confidence": 0.0,
                "safety_ratings": [],
                "error": str(e)[:300]
            }

    @retry_on_error(max_retries=2, delay=1)
    def chat_with_context(
        self,
        message: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced chat with context, quota detection, and educational focus
        
        Args:
            message: User's question
            context: Retrieved content from RAG system
            chat_history: Previous conversation (optional)
            
        Returns:
            Dict with response, confidence, and error flags
        """
        try:
            # Start new chat session
            chat = self.chat_model.start_chat(history=[])
            
            # Build educational prompt with context
            if context:
                system_prompt = f"""You are an AI teaching assistant for Physical AI and Humanoid Robotics.

CRITICAL INSTRUCTIONS:
1. Base your answers STRICTLY on the provided textbook content
2. If information isn't in the content, clearly state: "This specific topic isn't covered in the provided chapter sections"
3. Be educational, clear, and precise
4. Use examples and explanations from the book
5. Structure responses with clear sections when appropriate
6. Cite chapter references naturally (e.g., "According to Chapter X...")

TEXTBOOK CONTENT:
{context}

Now answer the student's question clearly and educationally:"""
                full_message = f"{system_prompt}\n\n{message}"
            else:
                full_message = message
            
            # Replay chat history (last 5 messages for context)
            if chat_history:
                for item in chat_history[-5:]:
                    if item.get("role") == "user" and item.get("content"):
                        try:
                            chat.send_message(item["content"])
                        except Exception as history_error:
                            logger.warning(f"Failed to replay history: {str(history_error)}")
                            pass
            
            # Send current message
            response = chat.send_message(full_message)
            
            # Check if response was blocked
            if not response.text:
                logger.warning("Chat response blocked by safety filters")
                return {
                    "response": "I apologize, but I cannot provide a response to this query. Please try rephrasing your question.",
                    "confidence": 0.0,
                    "safety_ratings": self._extract_safety_ratings(response),
                    "blocked": True
                }
            
            response_text = response.text.strip()
            confidence = self._calculate_confidence(response_text, has_context=bool(context))
            
            return {
                "response": response_text,
                "confidence": confidence,
                "safety_ratings": self._extract_safety_ratings(response),
                "blocked": False
            }
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error in chat: {str(e)}")
            logger.error(traceback.format_exc())
            
            # ✅ CRITICAL: Detect quota exceeded
            if any(keyword in error_str for keyword in ["quota", "429", "resourceexhausted"]):
                logger.warning("WARN: QUOTA EXCEEDED in chat_with_context")
                return {
                    "response": "[QUOTA_EXCEEDED]",
                    "confidence": 0.0,
                    "safety_ratings": [],
                    "quota_exceeded": True,
                    "error": str(e)[:300]
                }
            
            # Generic error
            return {
                "response": "I encountered an error while processing your message. Please try again.",
                "confidence": 0.0,
                "safety_ratings": [],
                "error": str(e)[:300]
            }

    @retry_on_error(max_retries=2, delay=1)
    def embed_content(self, text: str) -> List[float]:
        """
        Generate embeddings for text with validation and retry logic
        """
        try:
            # Validation
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for embedding")
                return []
            
            # Truncate if too long
            if len(text) > 10000:
                logger.warning(f"Text too long ({len(text)} chars), truncating to 10000")
                text = text[:10000]
            
            # Generate embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            
            return result["embedding"]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

    def _build_prompt(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """Build enhanced prompt with optional context and instructions"""
        parts = []
        
        if system_instruction:
            parts.append(system_instruction)
        
        if context:
            parts.append(f"Context from textbook:\n{context}\n")
        
        parts.append(f"Question: {prompt}")
        
        return "\n\n".join(parts)

    def _calculate_confidence(
        self, 
        text: str, 
        has_context: bool = False
    ) -> float:
        """
        Calculate confidence score based on response characteristics
        
        Factors:
        - Response length (longer = more confident, up to a point)
        - Presence of context (RAG-based = higher confidence)
        - Uncertainty phrases (lower confidence)
        """
        if not text:
            return 0.0
        
        word_count = len(text.split())
        
        # Base confidence from length
        if word_count < 10:
            base_confidence = 0.3
        elif word_count < 30:
            base_confidence = 0.6
        elif word_count < 100:
            base_confidence = 0.8
        else:
            base_confidence = 0.9
        
        # Boost confidence if we have context (RAG-based)
        if has_context:
            base_confidence = min(base_confidence + 0.15, 1.0)
        
        # Reduce confidence for uncertainty phrases
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i apologize",
            "not covered",
            "no information",
            "quota exceeded",
            "isn't covered"
        ]
        
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in uncertainty_phrases):
            base_confidence = max(base_confidence - 0.3, 0.1)
        
        return round(base_confidence, 2)

    def _extract_safety_ratings(self, response) -> List[Dict[str, Any]]:
        """Extract safety ratings from Gemini response"""
        try:
            if (hasattr(response, 'candidates') and 
                response.candidates and 
                len(response.candidates) > 0):
                
                return [
                    {
                        "category": (rating.category.name 
                                   if hasattr(rating.category, 'name') 
                                   else str(rating.category)),
                        "probability": (rating.probability.name 
                                      if hasattr(rating.probability, 'name') 
                                      else str(rating.probability))
                    }
                    for rating in response.candidates[0].safety_ratings
                ]
            return []
        except Exception as e:
            logger.warning(f"Could not extract safety ratings: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """Test if Gemini API is accessible and working"""
        try:
            test_response = self.generate_content("Hello, test connection.")
            return (bool(test_response.get("text")) and 
                   not test_response.get("error") and
                   test_response.get("text") != "[QUOTA_EXCEEDED]")
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
gemini_client = GeminiClient()


# ============================================================================
# SELF-TEST (when run directly)
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔧 Testing Gemini Client...")
    print("="*70)
    
    # Test 1: Connection
    print("\n1. Testing connection...")
    if gemini_client.test_connection():
        print("   SUCCESS: Connection successful!")
    else:
        print("   ERROR: Connection failed!")

    # Test 2: Simple generation
    print("\n2. Testing content generation...")
    result = gemini_client.generate_content("What is ROS 2?")
    if result.get("text") and result["text"] != "[QUOTA_EXCEEDED]":
        print(f"   SUCCESS: Generated {len(result['text'])} chars")
        print(f"   Confidence: {result.get('confidence', 0)}")
    else:
        print(f"   WARN: Generation issue: {result.get('error', 'Unknown')}")

    # Test 3: Chat with context
    print("\n3. Testing chat with context...")
    test_context = "ROS 2 is the Robot Operating System version 2, a middleware for robotics."
    chat_result = gemini_client.chat_with_context(
        "What is ROS 2?",
        context=test_context
    )
    if chat_result.get("response") and chat_result["response"] != "[QUOTA_EXCEEDED]":
        print(f"   SUCCESS: Chat response: {len(chat_result['response'])} chars")
        print(f"   Confidence: {chat_result.get('confidence', 0)}")
    else:
        print(f"   WARN: Chat issue: {chat_result.get('error', 'Unknown')}")

    # Test 4: Embeddings
    print("\n4. Testing embeddings...")
    embedding = gemini_client.embed_content("Test embedding generation")
    if embedding and len(embedding) > 0:
        print(f"   SUCCESS: Embedding generated: {len(embedding)} dimensions")
    else:
        print("   WARN: Embedding generation failed")

    print("\n" + "="*70)
    print("SUCCESS: All tests complete!")
    print("="*70 + "\n")