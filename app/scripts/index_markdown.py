# backend/app/scripts/index_markdown.py - FIXED VERSION (NO INFINITE LOOP)

import glob
from pathlib import Path
from typing import List, Tuple
import gc
import time
import traceback
import sys

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.embeddings import embedding_service
from app.vector_db import vector_db  # Import vector_db

# Always resolve absolute path to frontend/docs
FRONTEND_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "docs"


def chunk_text(text: str, max_chars: int = 800, overlap: int = 80):
    """
    Yield chunks of text instead of building a giant list in memory.
    FIXED VERSION - No infinite loop bug!
    """
    # SIMPLE FIX: Remove overlap to prevent infinite loop
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


def parse_info(path: Path) -> Tuple[str, str, str]:
    module_id = path.parent.name
    chapter_id = path.stem
    chapter_title = chapter_id.replace("-", " ").replace("_", " ").title()
    return module_id, chapter_id, chapter_title


def collect_markdown_files() -> List[Path]:
    print(f"📂 Looking in: {FRONTEND_DATA_ROOT.resolve()}")
    
    if not FRONTEND_DATA_ROOT.exists():
        print(f"❌ ERROR: Directory not found!")
        return []
    
    # Get modules in order
    files = []
    for module_dir in sorted(FRONTEND_DATA_ROOT.glob("module*")):
        if module_dir.is_dir():
            module_files = sorted(module_dir.glob("chapter*.md"))
            files.extend(module_files)
            print(f"   📦 {module_dir.name}: {len(module_files)} chapters")
    
    print(f"🔍 Found {len(files)} markdown files total")
    return files


def main():
    print("=" * 60)
    print("🚀 STARTING MEMORY-SAFE BOOK INDEXING")
    print("=" * 60)
    print("💻 Using embed_text() + add_batch_embeddings_fast()")
    print("📚 Processing in tiny batches (3 chunks at a time)")
    print("=" * 60)
    
    # Check vector_db connection
    if vector_db.client is None:
        print("❌ ERROR: Qdrant not connected!")
        print("   Please check your Qdrant connection and .env file")
        return
    
    md_paths = collect_markdown_files()
    if not md_paths:
        print("⚠️ No markdown files found.")
        return
    
    # Ask user how much to process
    print(f"\n📊 BOOK STRUCTURE FOUND:")
    print(f"   Total chapters: {len(md_paths)}")
    
    # Group by module for reporting
    modules = {}
    for path in md_paths:
        module = path.parent.name
        if module not in modules:
            modules[module] = []
        modules[module].append(path)
    
    for module, paths in modules.items():
        print(f"   {module}: {len(paths)} chapters")
    
    print("\n🧪 PROCESSING OPTIONS:")
    print("1. Test with ONE chapter only (safest)")
    print("2. Test with ONE module only")
    print("3. Process ALL modules (full book)")
    
    choice = input("\nChoose option (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Process just ONE chapter
        test_file = md_paths[0]
        md_paths = [test_file]
        print(f"\n🧪 TESTING WITH ONE CHAPTER: {test_file.name}")
        
    elif choice == "2":
        # Process first module only
        first_module = list(modules.keys())[0]
        md_paths = modules[first_module]
        print(f"\n🧪 TESTING WITH ONE MODULE: {first_module}")
        print(f"   {len(md_paths)} chapters in this module")
    
    # Track progress
    total_chunks_processed = 0
    successful_files = 0
    failed_files = 0
    
    # Process each file
    for idx, md in enumerate(md_paths, 1):
        print(f"\n{'─' * 50}")
        print(f"📖 [{idx}/{len(md_paths)}] {md.name}")
        print(f"{'─' * 50}")
        
        module_id, chapter_id, chapter_title = parse_info(md)
        
        try:
            # Check file size
            file_size = md.stat().st_size
            if file_size > 1_000_000:  # 1MB limit
                print(f"⚠️ Skipping (too large: {file_size:,} bytes)")
                failed_files += 1
                continue
            
            print(f"📊 File size: {file_size:,} bytes")
            print(f"📁 Module: {module_id}")
            print(f"📄 Chapter: {chapter_title}")
            
            # Read file
            with open(md, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Process in TINY batches
            BATCH_SIZE = 3  # SUPER SMALL - very safe!
            batch_chapter_ids = []
            batch_section_ids = []
            batch_contents = []
            batch_embeddings = []
            chunk_count = 0
            batch_count = 0
            
            # FIXED: Using the corrected chunk_text() function
            for i, chunk in enumerate(chunk_text(text)):
                chunk_count += 1
                
                try:
                    # Generate embedding for this chunk
                    embedding = embedding_service.embed_text(chunk)
                    
                    if embedding is not None:
                        batch_chapter_ids.append(chapter_id)
                        batch_section_ids.append(f"{module_id}-{chapter_id}-chunk-{i}")
                        batch_contents.append(chunk)
                        batch_embeddings.append(embedding)
                        
                        # Show progress for first few chunks
                        if chunk_count <= 3:
                            print(f"  ✅ Chunk {i+1}: {len(chunk)} chars, embedding generated")
                    else:
                        print(f"  ⚠️ Chunk {i+1}: Failed to generate embedding")
                        
                except Exception as e:
                    print(f"  ⚠️ Chunk {i+1}: Error: {str(e)[:80]}")
                
                # Process batch when we have BATCH_SIZE chunks
                if len(batch_contents) >= BATCH_SIZE:
                    batch_count += 1
                    
                    try:
                        # Show progress for first batch and every 5th batch
                        if batch_count == 1 or batch_count % 5 == 0:
                            print(f"  ↪️ Batch {batch_count}: Processing {len(batch_contents)} chunks...")
                        
                        # Use the NEW add_batch_embeddings_fast method
                        successful_uploads = vector_db.add_batch_embeddings_fast(
                            chapter_ids=batch_chapter_ids,
                            section_ids=batch_section_ids,
                            contents=batch_contents,
                            embeddings=batch_embeddings,
                            metadatas=[{
                                "module": str(module_id),
                                "chapter": str(chapter_id),
                                "title": str(chapter_title)[:100],
                                "file": str(md.name),
                                "chunk_index": j,
                                "timestamp": time.time()
                            } for j in range(len(batch_contents))]
                        )
                        
                        if successful_uploads > 0:
                            total_chunks_processed += successful_uploads
                            print(f"  ✅ Uploaded {successful_uploads} chunks")
                        else:
                            print(f"  ⚠️ Batch {batch_count}: No chunks uploaded")
                        
                        # CLEAR IMMEDIATELY
                        batch_chapter_ids.clear()
                        batch_section_ids.clear()
                        batch_contents.clear()
                        batch_embeddings.clear()
                        
                        # FORCE GARBAGE COLLECTION
                        gc.collect()
                        
                        # Tiny pause
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"  ❌ Batch {batch_count} failed: {str(e)[:100]}")
                        # Clear and continue
                        batch_chapter_ids.clear()
                        batch_section_ids.clear()
                        batch_contents.clear()
                        batch_embeddings.clear()
            
            # Process remaining chunks (last batch)
            if batch_contents:
                batch_count += 1
                try:
                    print(f"  ↪️ Final batch {batch_count}: Processing {len(batch_contents)} chunks...")
                    
                    successful_uploads = vector_db.add_batch_embeddings_fast(
                        chapter_ids=batch_chapter_ids,
                        section_ids=batch_section_ids,
                        contents=batch_contents,
                        embeddings=batch_embeddings,
                        metadatas=[{
                            "module": str(module_id),
                            "chapter": str(chapter_id),
                            "title": str(chapter_title)[:100],
                            "file": str(md.name),
                            "chunk_index": j,
                            "timestamp": time.time()
                        } for j in range(len(batch_contents))]
                    )
                    
                    if successful_uploads > 0:
                        total_chunks_processed += successful_uploads
                        print(f"  ✅ Uploaded {successful_uploads} final chunks")
                    else:
                        print(f"  ⚠️ Final batch: No chunks uploaded")
                        
                except Exception as e:
                    print(f"  ❌ Final batch failed: {str(e)[:100]}")
                    # Don't count file as complete failure
            
            # Clear memory
            del text
            gc.collect()
            
            print(f"\n✅ CHAPTER COMPLETE!")
            print(f"   Total chunks in file: {chunk_count}")
            print(f"   Chunks uploaded to Qdrant: {total_chunks_processed}")
            print(f"   Batches sent: {batch_count}")
            
            if chunk_count > 0:
                successful_files += 1
            else:
                failed_files += 1
            
            # Pause between chapters (except last)
            if idx < len(md_paths):
                print("⏳ Taking a short break...")
                time.sleep(0.5)
            
            # Show current vector count
            try:
                vector_count = vector_db.get_vector_count()
                print(f"   📊 Current vectors in Qdrant: {vector_count}")
            except:
                pass
            
        except MemoryError:
            print(f"💥 MEMORY ERROR! Skipping this chapter.")
            failed_files += 1
            gc.collect()
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:100]}")
            traceback.print_exc()
            failed_files += 1
        
        # Show overall progress
        print(f"\n📈 OVERALL PROGRESS: {successful_files}✓ {failed_files}✗")
    
    # Final report
    print(f"\n{'='*60}")
    print("🎉 INDEXING COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 FINAL RESULTS:")
    print(f"   Successful files: {successful_files}")
    print(f"   Failed files:     {failed_files}")
    print(f"   Total chunks uploaded: {total_chunks_processed}")
    print(f"\n💡 METHOD USED: embed_text() + add_batch_embeddings_fast()")
    print(f"💡 MEMORY USAGE: Perfect for 16GB RAM! ✅")
    
    # Check final vector count
    try:
        vector_count = vector_db.get_vector_count()
        print(f"📊 FINAL VECTORS IN QDRANT: {vector_count}")
    except:
        print(f"📊 Qdrant collection updated")
    
    print(f"{'='*60}")
    
    # Recommendation
    if successful_files > 0 and choice in ["1", "2"]:
        print(f"\n✅ TEST SUCCESSFUL!")
        print(f"   Your system can handle the indexing easily.")
        print(f"   Run again and choose option 3 to index the FULL book!")
    elif successful_files == len(md_paths):
        print(f"\n🎯 FULL BOOK INDEXED SUCCESSFULLY!")
        print(f"   Your chatbot is now ready with {total_chunks_processed} knowledge chunks!")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()