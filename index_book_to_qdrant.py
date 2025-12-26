# index_book_to_qdrant.py - Upload Docusaurus Content with Enhanced Metadata

"""
This script reads your Docusaurus book content and uploads it to Qdrant
with RICH METADATA for accurate chapter-specific retrieval.

USAGE:
1. Update DOCS_PATH to point to your Docusaurus docs folder
2. Run: python index_book_to_qdrant.py
3. Wait for all chapters to be indexed

The script will:
- Read all markdown files from your Docusaurus docs
- Extract chapter structure (module, number, title)
- Chunk content intelligently (by headings)
- Generate embeddings
- Upload to Qdrant with rich metadata
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Import your services
from app.embeddings import embedding_service
from app.vector_db import vector_db


# ==================== CONFIGURATION ====================

# Update this to your Docusaurus docs folder
DOCS_PATH = "C:/Users/WWW.SZLAIWIIT.COM/specify_plus/ai_and_robot_book4/frontend/docs"  # Change this to your actual path

# Module mapping (based on your 4-module structure)
MODULE_MAPPING = {
    "module1": "Module 1: ROS 2 Fundamentals",
    "module2": "Module 2: Gazebo & Unity Simulation",
    "module3": "Module 3: NVIDIA Isaac Platform",
    "module4": "Module 4: Vision-Language-Action Models"
}

# ==================== HELPER FUNCTIONS ====================

def parse_markdown_file(filepath: str) -> Dict:
    """
    Parse markdown file and extract metadata
    Returns: {
        'content': full content,
        'title': chapter title,
        'frontmatter': metadata from frontmatter
    }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract frontmatter (between --- and ---)
    frontmatter = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            content = parts[2]
            
            # Simple frontmatter parsing
            for line in fm_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
    
    # Extract title from frontmatter or first heading
    title = frontmatter.get('title', '')
    if not title:
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            title = match.group(1)
    
    return {
        'content': content.strip(),
        'title': title,
        'frontmatter': frontmatter
    }


def extract_chapter_info_from_path(filepath: str) -> Dict:
    """
    Extract module and chapter info from file path
    
    Example paths:
    - docs/module1/chapter01.md → Module 1, Chapter 1
    - docs/module4/chapter35.md → Module 4, Chapter 35
    - docs/module1/chapter1-1-intro.md → Module 1, Chapter 1.1
    - docs/module2/week3/gazebo-basics.md → Module 2, Chapter 2.3
    """
    path_parts = Path(filepath).parts
    
    # Find module
    module = "General"
    module_number = 0
    for part in path_parts:
        if part.startswith('module'):
            module_num = re.search(r'module(\d+)', part)
            if module_num:
                module_number = int(module_num.group(1))
                module = MODULE_MAPPING.get(part, f"Module {module_number}")
                break
    
    # Extract chapter number from filename or path
    filename = Path(filepath).stem
    chapter_number = ""
    
    # Try patterns like: chapter35, chapter1-1, 1.1, week1, etc.
    # ✅ FIXED: Added pattern for single chapter numbers (chapter35, chapter01)
    patterns = [
        r'chapter(\d+)',            # chapter35, chapter01 - SINGLE NUMBER (MOST COMMON!)
        r'chapter(\d+)[_-](\d+)',   # chapter1-1, chapter1_1
        r'(\d+)\.(\d+)',            # 1.1, 2.3
        r'week(\d+)',               # week1, week2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            groups = match.groups()
            # ✅ FIXED: Handle single number properly
            if len(groups) == 1:
                chapter_number = groups[0]  # Just use the number: "35"
            elif len(groups) == 2:
                chapter_number = f"{groups[0]}.{groups[1]}"  # Use format: "1.1"
            break
    
    return {
        'module': module,
        'module_number': module_number,
        'chapter_number': chapter_number
    }


def chunk_content_by_headings(content: str, max_chunk_size: int = 1000) -> List[Tuple[str, str]]:
    """
    Smart chunking: Split content by headings but keep sections together
    Returns: [(section_title, section_content), ...]
    """
    chunks = []
    
    # Split by headings (## or ###)
    sections = re.split(r'^(#{2,3}\s+.+)$', content, flags=re.MULTILINE)
    
    current_heading = "Introduction"
    current_content = ""
    
    for i, section in enumerate(sections):
        if section.strip().startswith('##'):
            # This is a heading
            if current_content.strip():
                # Save previous section
                chunks.append((current_heading, current_content.strip()))
            
            current_heading = section.strip('#').strip()
            current_content = ""
        else:
            # This is content
            current_content += section
            
            # If content gets too long, split it
            if len(current_content) > max_chunk_size:
                chunks.append((current_heading, current_content.strip()))
                current_heading = f"{current_heading} (continued)"
                current_content = ""
    
    # Add last section
    if current_content.strip():
        chunks.append((current_heading, current_content.strip()))
    
    return chunks


def determine_content_type(heading: str, content: str) -> str:
    """Determine the type of content"""
    heading_lower = heading.lower()
    content_lower = content.lower()
    
    if any(word in heading_lower for word in ['introduction', 'overview', 'what is']):
        return "introduction"
    elif any(word in heading_lower for word in ['tutorial', 'how to', 'step by step', 'guide']):
        return "tutorial"
    elif any(word in heading_lower for word in ['example', 'demo', 'sample']):
        return "example"
    elif any(word in heading_lower for word in ['exercise', 'practice', 'assignment']):
        return "exercise"
    elif '```' in content or 'def ' in content or 'class ' in content:
        return "code_example"
    else:
        return "general"


# ==================== MAIN INDEXING FUNCTION ====================

def index_docusaurus_book():
    """Main function to index entire Docusaurus book"""
    
    print("=" * 70)
    print("📚 INDEXING DOCUSAURUS BOOK TO QDRANT")
    print("=" * 70)
    
    if not os.path.exists(DOCS_PATH):
        print(f"❌ Error: Docs path not found: {DOCS_PATH}")
        print("   Please update DOCS_PATH in this script")
        return
    
    # Find all markdown files
    md_files = list(Path(DOCS_PATH).rglob("*.md"))
    print(f"\n📄 Found {len(md_files)} markdown files")
    
    if not md_files:
        print("❌ No markdown files found!")
        return
    
    total_chunks = 0
    successful_uploads = 0
    
    # Process each file
    for file_idx, md_file in enumerate(md_files, 1):
        print(f"\n[{file_idx}/{len(md_files)}] Processing: {md_file.name}")
        
        try:
            # Parse markdown
            parsed = parse_markdown_file(str(md_file))
            chapter_info = extract_chapter_info_from_path(str(md_file))
            
            print(f"   📖 Title: {parsed['title']}")
            print(f"   📚 Module: {chapter_info['module']}")
            print(f"   🔢 Chapter: {chapter_info['chapter_number'] or 'N/A'}")
            
            # Chunk content
            chunks = chunk_content_by_headings(parsed['content'])
            print(f"   ✂️  Split into {len(chunks)} sections")
            
            # Process each chunk
            for chunk_idx, (section_title, section_content) in enumerate(chunks):
                if len(section_content.strip()) < 50:
                    continue  # Skip very short chunks
                
                # Generate embedding
                embedding = embedding_service.embed_text(section_content)
                if not embedding:
                    print(f"      ⚠️  Skipping section '{section_title[:50]}...' (embedding failed)")
                    continue
                
                # Build metadata
                metadata = {
                    'chapter_number': chapter_info['chapter_number'],
                    'chapter_title': parsed['title'],
                    'module': chapter_info['module'],
                    'module_number': chapter_info['module_number'],
                    'section_title': section_title,
                    'subsection': f"Section {chunk_idx + 1}",
                    'content_type': determine_content_type(section_title, section_content),
                    'difficulty': 'intermediate',  # You can make this smarter
                    'file_path': str(md_file.relative_to(DOCS_PATH))
                }
                
                # Upload to Qdrant
                success = vector_db.add_embeddings(
                    chapter_id=f"{chapter_info['module_number']}-{chapter_info['chapter_number']}",
                    section_id=section_title,
                    content=section_content,
                    embedding=embedding,
                    metadata=metadata
                )
                
                if success:
                    successful_uploads += 1
                    total_chunks += 1
                
                # Progress indicator
                if successful_uploads % 10 == 0:
                    print(f"      ✅ Uploaded {successful_uploads} chunks so far...")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error processing {md_file.name}: {str(e)[:100]}")
            continue
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ INDEXING COMPLETE!")
    print("=" * 70)
    print(f"📊 Total files processed: {len(md_files)}")
    print(f"📦 Total chunks created: {total_chunks}")
    print(f"✅ Successfully uploaded: {successful_uploads}")
    print(f"📈 Qdrant now has: {vector_db.get_vector_count()} vectors")
    print("=" * 70)
    print("\n🎉 All done! Your chatbot should now have accurate chapter-specific responses!")


# ==================== RUN ====================

if __name__ == "__main__":
    print("\n🚀 Starting book indexing process...")
    print("⚠️  Make sure your .env file has correct Qdrant credentials!")
    
    # Test connection first
    if not vector_db.test_connection():
        print("\n❌ Qdrant connection failed! Please check your .env file")
        exit(1)
    
    print("✅ Qdrant connection successful!\n")
    
    # Run indexing
    index_docusaurus_book()
    
    print("\n🎉 Chapter numbers should now be properly extracted!")