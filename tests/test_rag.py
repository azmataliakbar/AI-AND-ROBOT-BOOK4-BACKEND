import pytest
from unittest.mock import MagicMock, patch
from app.rag import RAGService
from app.schemas import ChatRequest


@pytest.fixture
def rag_service():
    """Create a RAG service instance for testing"""
    with patch('app.rag.embedding_service'), \
         patch('app.rag.gemini_client'):
        service = RAGService()
        yield service


def test_get_response_with_vector_results(rag_service):
    """Test that get_response returns vector results when available"""
    # Mock vector search results
    mock_vector_results = [
        {
            "chapter_id": "ch_001",
            "content": "This is a test content for ROS 2",
            "similarity_score": 0.85,
            "section_id": "section_1"
        }
    ]

    with patch.object(rag_service.embedding_service, 'search_similar_content', return_value=mock_vector_results), \
         patch.object(rag_service.gemini_client, 'chat_with_context', return_value={
             "response": "This is a test response",
             "confidence": 0.85,
             "safety_ratings": []
         }):

        result = rag_service.get_response("test query", "user_123")

        assert result.response == "This is a test response"
        assert result.confidence_score == 0.85
        assert result.source_type == "vector"


def test_get_response_fallback_when_no_vector_results(rag_service):
    """Test that get_response falls back to Gemini when no vector results"""
    # Mock no vector search results
    mock_vector_results = []

    with patch.object(rag_service.embedding_service, 'search_similar_content', return_value=mock_vector_results), \
         patch.object(rag_service.gemini_client, 'generate_content', return_value={
             "text": "This is a fallback response",
             "confidence": 0.6,
             "safety_ratings": []
         }):

        result = rag_service.get_response("test query", "user_123")

        assert result.response == "This is a fallback response"
        assert result.confidence_score == 0.6
        assert result.source_type == "fallback"


def test_get_search_results(rag_service):
    """Test that search results are returned correctly"""
    mock_vector_results = [
        {
            "chapter_id": "ch_001",
            "content": "This is a test content for ROS 2",
            "similarity_score": 0.85,
            "section_id": "section_1",
            "chapter_title": "Introduction to ROS 2"
        }
    ]

    with patch.object(rag_service.embedding_service, 'search_similar_content', return_value=mock_vector_results):
        results = rag_service.get_search_results("test query")

        assert len(results) == 1
        assert results[0].chapter_id == "ch_001"
        assert results[0].similarity_score == 0.85