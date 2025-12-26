# File: check_qdrant.py
import sys
sys.path.append('.')
from app.vector_db import vector_db

# Check what's in Qdrant
print("🔍 CHECKING QDRANT CONTENTS:")
print(f"Total vectors: {vector_db.get_vector_count()}")

# Get some sample vectors
sample_vectors = vector_db.client.scroll(
    collection_name="robotics_book",
    limit=5,
    with_payload=True  # Include the actual text content
)

print("\n📄 SAMPLE CHUNKS IN QDRANT:")
for i, point in enumerate(sample_vectors[0], 1):
    print(f"\n--- Chunk {i} ---")
    print(f"ID: {point.id}")
    print(f"Chapter: {point.payload.get('chapter', 'N/A')}")
    print(f"Module: {point.payload.get('module', 'N/A')}")
    print(f"Content (first 100 chars): {point.payload.get('content', '')[:100]}...")

# Search for something
print("\n🔎 TEST SEARCH:")
results = vector_db.search_similar_content("What is ROS 2?", limit=3)
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Chapter: {result.get('chapter', 'N/A')}")
    print(f"  Score: {result.get('score', 0):.3f}")
    print(f"  Content: {result.get('content', '')[:80]}...")