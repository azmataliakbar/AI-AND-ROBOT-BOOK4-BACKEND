# create_qdrant_indexes.py - Add Payload Indexes to Qdrant Collection

"""
This script creates payload indexes for chapter_number and topics fields
in your Qdrant collection to enable efficient filtering.

USAGE:
Run: python create_qdrant_indexes.py
"""

from app.vector_db import vector_db
from qdrant_client.models import PayloadSchemaType

print("=" * 70)
print("🔧 CREATING QDRANT PAYLOAD INDEXES")
print("=" * 70)

# Test connection
print("\n📡 Testing Qdrant connection...")
if not vector_db.test_connection():
    print("❌ Qdrant connection failed!")
    exit(1)

print("✅ Connected to Qdrant")
print(f"📚 Collection: {vector_db.collection_name}")
print(f"📊 Current vectors: {vector_db.get_vector_count()}")

# Create indexes
print("\n🔨 Creating payload indexes...")

try:
    # Index 1: chapter_number (keyword/string)
    print("\n[1/3] Creating index for 'chapter_number' (keyword)...")
    vector_db.client.create_payload_index(
        collection_name=vector_db.collection_name,
        field_name="chapter_number",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✅ chapter_number index created!")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print("   ⚠️  Index already exists (skipping)")
    else:
        print(f"   ❌ Error: {str(e)[:100]}")

try:
    # Index 2: module_number (integer)
    print("\n[2/3] Creating index for 'module_number' (integer)...")
    vector_db.client.create_payload_index(
        collection_name=vector_db.collection_name,
        field_name="module_number",
        field_schema=PayloadSchemaType.INTEGER
    )
    print("   ✅ module_number index created!")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print("   ⚠️  Index already exists (skipping)")
    else:
        print(f"   ❌ Error: {str(e)[:100]}")

try:
    # Index 3: topics (keyword array)
    print("\n[3/3] Creating index for 'topics' (keyword)...")
    vector_db.client.create_payload_index(
        collection_name=vector_db.collection_name,
        field_name="topics",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✅ topics index created!")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print("   ⚠️  Index already exists (skipping)")
    else:
        print(f"   ❌ Error: {str(e)[:100]}")

# Verify indexes
print("\n" + "=" * 70)
print("✅ INDEX CREATION COMPLETE!")
print("=" * 70)
print("\n📋 Summary:")
print("   • chapter_number: Keyword index for filtering chapters")
print("   • module_number: Integer index for filtering modules")
print("   • topics: Keyword index for topic-based filtering")
print("\n💡 Next step: Restart your backend to test!")
print("=" * 70)
