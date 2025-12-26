# clear_qdrant.py
# Script to clear all old data from Qdrant before re-indexing with new metadata

from app.vector_db import vector_db

print("\n" + "=" * 70)
print("🗑️  CLEARING OLD QDRANT DATA")
print("=" * 70)

# Check current status
current_count = vector_db.get_vector_count()
print(f"\n📊 Current vector count: {current_count}")

if current_count == 0:
    print("✅ Qdrant is already empty - nothing to clear!")
    print("=" * 70)
    exit(0)

# Get all chapter IDs
all_chapters = vector_db.get_all_chapter_ids()
print(f"📚 Found {len(all_chapters)} chapters to delete")

if len(all_chapters) == 0:
    print("⚠️  No chapters found, but vectors exist. Trying alternative method...")
    # If no chapters found, the collection might be corrupted or use different structure
    print("   Recommendation: Delete collection manually in Qdrant Cloud dashboard")
    print("=" * 70)
    exit(1)

# Delete each chapter
print("\n🔄 Starting deletion process...")
total_deleted = 0

for i, chapter_id in enumerate(all_chapters, 1):
    try:
        deleted = vector_db.delete_by_chapter_id(chapter_id)
        total_deleted += deleted
        print(f"   [{i}/{len(all_chapters)}] ✅ Deleted {deleted} vectors for chapter: {chapter_id}")
    except Exception as e:
        print(f"   [{i}/{len(all_chapters)}] ❌ Error deleting chapter {chapter_id}: {str(e)[:100]}")

# Check final status
final_count = vector_db.get_vector_count()

print("\n" + "=" * 70)
print("✅ DELETION COMPLETE!")
print("=" * 70)
print(f"📊 Vectors deleted: {total_deleted}")
print(f"📊 Initial count: {current_count}")
print(f"📊 Final count: {final_count}")

if final_count == 0:
    print("\n🎉 SUCCESS! Qdrant is now empty and ready for re-indexing.")
    print("👉 Next step: Run 'python index_book_to_qdrant.py'")
else:
    print(f"\n⚠️  WARNING: {final_count} vectors still remain")
    print("   Some vectors might not have chapter_id metadata")
    print("   You may need to manually delete the collection in Qdrant Cloud")

print("=" * 70 + "\n")

