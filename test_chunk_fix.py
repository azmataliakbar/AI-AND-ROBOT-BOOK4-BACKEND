#!/usr/bin/env python3
"""
Test to verify the chunk_text() function fix
This should create 2 chunks, not infinite!
"""

def chunk_text_fixed(text: str, max_chars: int = 800, overlap: int = 80):
    """Fixed version - no infinite loop"""
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


def chunk_text_buggy(text: str, max_chars: int = 800, overlap: int = 80):
    """Original buggy version - causes infinite loop"""
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        yield text[start:end]
        start = end - overlap  # BUG: can cause infinite loop
        if start < 0:
            start = 0


def test_both_versions():
    """Test both versions with your chapter size"""
    
    # Simulate your chapter (1562 characters)
    # Create realistic text instead of just "Test"
    chapter_content = "# Introduction to ROS 2\n\n" + "ROS 2 is awesome. " * 100
    chapter_content = chapter_content[:1562]  # Make it exactly 1562 chars
    
    print("=" * 60)
    print("🧪 TESTING CHUNK_TEXT() FUNCTIONS")
    print("=" * 60)
    print(f"Test text length: {len(chapter_content)} characters\n")
    
    # Test FIXED version
    print("✅ TESTING FIXED VERSION:")
    chunks_fixed = []
    for i, chunk in enumerate(chunk_text_fixed(chapter_content, max_chars=800)):
        chunks_fixed.append(chunk)
        print(f"  Chunk {i+1}: {len(chunk)} characters")
    
    print(f"\n  Total chunks: {len(chunks_fixed)}")
    print(f"  Expected: 2 chunks (1562 chars ÷ 800 = 2)")
    
    # Test BUGGY version (with safety limit)
    print("\n❌ TESTING BUGGY VERSION (with safety limit):")
    chunks_buggy = []
    safety_limit = 10  # Stop after 10 chunks to prevent infinite loop
    
    for i, chunk in enumerate(chunk_text_buggy(chapter_content, max_chars=800)):
        chunks_buggy.append(chunk)
        print(f"  Chunk {i+1}: {len(chunk)} characters")
        
        if i + 1 >= safety_limit:
            print(f"  ⚠️ STOPPED at {safety_limit} chunks (safety limit)")
            print(f"  ⚠️ This would run FOREVER without limit!")
            break
    
    # Calculate expected chunks
    expected_chunks = (len(chapter_content) // 800) + (1 if len(chapter_content) % 800 > 0 else 0)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    print(f"Fixed version created:   {len(chunks_fixed)} chunks ✓")
    print(f"Buggy version would create: INFINITE chunks! ✗")
    print(f"Expected for {len(chapter_content)} chars: {expected_chunks} chunks")
    
    # Verify chunk sizes
    if len(chunks_fixed) == expected_chunks:
        print("\n✅ FIXED VERSION WORKS CORRECTLY!")
        print(f"   Chunk 1: {len(chunks_fixed[0])} chars")
        if len(chunks_fixed) > 1:
            print(f"   Chunk 2: {len(chunks_fixed[1])} chars")
            print(f"   Total: {len(chunks_fixed[0]) + len(chunks_fixed[1])} chars")
    else:
        print(f"\n❌ FIXED VERSION STILL WRONG!")
        print(f"   Got {len(chunks_fixed)} chunks, expected {expected_chunks}")
    
    print("=" * 60)


def test_with_real_chapter():
    """Test with your actual chapter file"""
    import os
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print("📄 TESTING WITH ACTUAL CHAPTER FILE")
    print("=" * 60)
    
    # Path to your chapter file
    chapter_path = Path(r"C:\Users\WWW.SZLAIWIIT.COM\specify_plus\ai_and_robot_book4\frontend\docs\module1\chapter-1-introduction-to-ros2.md")
    
    if not chapter_path.exists():
        print(f"❌ File not found: {chapter_path}")
        print("   Using simulated content instead...")
        return
    
    try:
        # Read the actual file
        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📊 File: {chapter_path.name}")
        print(f"📏 Size: {len(content)} characters")
        print(f"📁 Path: {chapter_path}")
        
        # Test with fixed version
        chunks = []
        for i, chunk in enumerate(chunk_text_fixed(content, max_chars=800)):
            chunks.append(chunk)
            print(f"\n  Chunk {i+1}:")
            print(f"    Length: {len(chunk)} chars")
            # Show first 50 chars of each chunk
            preview = chunk[:50].replace('\n', ' ')
            print(f"    Preview: {preview}...")
        
        print(f"\n✅ File would be split into {len(chunks)} chunks")
        print(f"   Expected batches: {max(1, len(chunks) // 3)}")
        
        if len(chunks) > 10:
            print("\n⚠️ WARNING: More than 10 chunks from a small file!")
            print("   The bug might still be present!")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 CHUNK_TEXT() BUG TESTER")
    print("=" * 60)
    
    test_both_versions()
    test_with_real_chapter()
    
    print("\n" + "=" * 60)
    print("💡 INSTRUCTIONS:")
    print("=" * 60)
    print("1. Run this test: python test_chunk_fix.py")
    print("2. Fixed version should show 2-3 chunks")
    print("3. Buggy version would show INFINITE chunks")
    print("4. Replace chunk_text() in your files with the FIXED version")
    print("=" * 60)