# populate_postgres.py - Populate PostgreSQL with Book Chapters

"""
This script reads your Docusaurus book content and populates PostgreSQL
with chapter metadata for the chatbot to use.

USAGE:
1. Update DOCS_PATH to point to your Docusaurus docs folder
2. Run: python populate_postgres.py
3. Chapters will be inserted into PostgreSQL

This complements index_book_to_qdrant.py which handles vector embeddings.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import hashlib

# Import your database
from app.database import SessionLocal, engine
from app.models import Chapter, Base
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text  # ✅ ADDED THIS LINE


# ==================== CONFIGURATION ====================

# Update this to your Docusaurus docs folder
DOCS_PATH = "C:/Users/WWW.SZLAIWIIT.COM/specify_plus/ai_and_robot_book4/frontend/docs"

# Module mapping
MODULE_MAPPING = {
    "module1": 1,
    "module2": 2,
    "module3": 3,
    "module4": 4
}

MODULE_NAMES = {
    1: "ROS 2 Fundamentals",
    2: "Gazebo & Unity Simulation", 
    3: "NVIDIA Isaac Platform",
    4: "Vision-Language-Action Models"
}


# ==================== HELPER FUNCTIONS ====================

def parse_markdown_file(filepath: str) -> Dict:
    """Parse markdown file and extract content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract frontmatter
    frontmatter = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            content = parts[2]
            
            for line in fm_text.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
    
    # Extract title
    title = frontmatter.get('title', '')
    if not title:
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            title = match.group(1)
    
    # Extract description (first paragraph or from frontmatter)
    description = frontmatter.get('description', '')
    if not description:
        # Get first paragraph
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        if paragraphs:
            description = paragraphs[0][:300]  # First 300 chars
    
    return {
        'content': content.strip(),
        'title': title,
        'description': description,
        'frontmatter': frontmatter
    }


def extract_chapter_info_from_path(filepath: str) -> Dict:
    """Extract module and chapter info from file path"""
    path_parts = Path(filepath).parts
    
    # Find module
    module_number = 0
    for part in path_parts:
        if part.startswith('module'):
            module_num = re.search(r'module(\d+)', part)
            if module_num:
                module_number = int(module_num.group(1))
                break
    
    # Extract chapter number from filename
    filename = Path(filepath).stem
    chapter_number = 0
    
    # Try patterns
    patterns = [
        r'chapter[_-]?(\d+)',       # chapter1, chapter-1, chapter_1
        r'^(\d+)[_-]',              # 1-, 1_
        r'week(\d+)',               # week1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            chapter_number = int(match.group(1))
            break
    
    # Calculate overall chapter number (e.g., Module 2 Chapter 3 = Chapter 13)
    overall_chapter = (module_number - 1) * 10 + chapter_number if module_number > 0 else chapter_number
    
    return {
        'module': module_number,
        'chapter_number': overall_chapter,
        'module_chapter': chapter_number  # Chapter within module
    }


def generate_chapter_id(module: int, chapter: int, title: str) -> str:
    """Generate consistent chapter ID"""
    # Use hash of title for uniqueness
    title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
    return f"ch_{module}_{chapter}_{title_hash}"


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def estimate_reading_time(word_count: int) -> int:
    """Estimate reading time in minutes (250 words/min)"""
    return max(1, word_count // 250)


# ==================== MAIN POPULATION FUNCTION ====================

def populate_postgres():
    """Main function to populate PostgreSQL with chapters"""
    
    print("=" * 70)
    print("📚 POPULATING POSTGRESQL WITH BOOK CHAPTERS")
    print("=" * 70)
    
    # Create tables if they don't exist
    print("\n🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables ready!")
    
    if not os.path.exists(DOCS_PATH):
        print(f"\n❌ Error: Docs path not found: {DOCS_PATH}")
        print("   Please update DOCS_PATH in this script")
        return
    
    # Find all markdown files
    md_files = list(Path(DOCS_PATH).rglob("*.md"))
    print(f"\n📄 Found {len(md_files)} markdown files")
    
    if not md_files:
        print("❌ No markdown files found!")
        return
    
    # Create database session
    db = SessionLocal()
    
    inserted_count = 0
    updated_count = 0
    skipped_count = 0
    
    try:
        # Process each file
        for file_idx, md_file in enumerate(md_files, 1):
            print(f"\n[{file_idx}/{len(md_files)}] Processing: {md_file.name}")
            
            try:
                # Parse markdown
                parsed = parse_markdown_file(str(md_file))
                chapter_info = extract_chapter_info_from_path(str(md_file))
                
                if not parsed['title']:
                    print(f"   ⚠️  Skipping: No title found")
                    skipped_count += 1
                    continue
                
                if chapter_info['module'] == 0:
                    print(f"   ⚠️  Skipping: Not in a module folder")
                    skipped_count += 1
                    continue
                
                # Generate chapter ID
                chapter_id = generate_chapter_id(
                    chapter_info['module'],
                    chapter_info['chapter_number'],
                    parsed['title']
                )
                
                # Count words and estimate time
                word_count = count_words(parsed['content'])
                reading_time = estimate_reading_time(word_count)
                
                print(f"   📖 Title: {parsed['title']}")
                print(f"   📚 Module: {chapter_info['module']} - {MODULE_NAMES.get(chapter_info['module'], 'Unknown')}")
                print(f"   🔢 Chapter: {chapter_info['chapter_number']}")
                print(f"   📝 Words: {word_count} (~{reading_time} min read)")
                
                # Check if chapter already exists
                existing = db.query(Chapter).filter(Chapter.id == chapter_id).first()
                
                if existing:
                    # Update existing chapter
                    existing.title = parsed['title']
                    existing.module = chapter_info['module']
                    existing.chapter_number = chapter_info['chapter_number']
                    existing.content = parsed['content']
                    existing.word_count = word_count
                    existing.estimated_reading_time = reading_time
                    
                    if parsed['description']:
                        existing.description = parsed['description']
                    
                    print(f"   🔄 Updated existing chapter")
                    updated_count += 1
                else:
                    # Create new chapter
                    new_chapter = Chapter(
                        id=chapter_id,
                        title=parsed['title'],
                        module=chapter_info['module'],
                        chapter_number=chapter_info['chapter_number'],
                        content=parsed['content'],
                        word_count=word_count,
                        estimated_reading_time=reading_time
                    )
                    
                    if parsed['description']:
                        new_chapter.description = parsed['description']
                    
                    db.add(new_chapter)
                    print(f"   ✅ Inserted new chapter")
                    inserted_count += 1
                
                # Commit after each chapter
                db.commit()
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}")
                db.rollback()
                skipped_count += 1
                continue
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ POPULATION COMPLETE!")
        print("=" * 70)
        print(f"📊 Total files processed: {len(md_files)}")
        print(f"➕ New chapters inserted: {inserted_count}")
        print(f"🔄 Existing chapters updated: {updated_count}")
        print(f"⏭️  Skipped: {skipped_count}")
        
        # Show final count
        total_chapters = db.query(Chapter).count()
        print(f"\n📚 Total chapters in database: {total_chapters}")
        
        # Show breakdown by module
        print("\n📊 Chapters by Module:")
        for module_num in [1, 2, 3, 4]:
            count = db.query(Chapter).filter(Chapter.module == module_num).count()
            module_name = MODULE_NAMES.get(module_num, f"Module {module_num}")
            print(f"   Module {module_num} ({module_name}): {count} chapters")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        db.rollback()
    finally:
        db.close()


# ==================== RUN ====================

if __name__ == "__main__":
    print("\n🚀 Starting PostgreSQL population...")
    print("⚠️  Make sure your .env file has correct DATABASE_URL!")
    
    # Test database connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))  # ✅ FIXED THIS LINE
        db.close()
        print("✅ Database connection successful!\n")
    except Exception as e:
        print(f"\n❌ Database connection failed: {str(e)[:100]}")
        print("   Please check your DATABASE_URL in .env")
        exit(1)
    
    # Run population
    populate_postgres()
    
    print("\n🎉 All done! PostgreSQL now has your chapters!")
    print("💡 Next: Run index_book_to_qdrant.py to update vector embeddings")