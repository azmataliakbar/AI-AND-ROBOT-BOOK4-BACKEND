# backend/app/scripts/index_markdown.py

import glob
from pathlib import Path
from typing import List, Tuple

from app.embeddings import embedding_service
from app.vector_db import vector_db

# Always resolve absolute path to frontend/docs
FRONTEND_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend" / "docs"


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Yield chunks of text instead of building a giant list in memory.
    Safer for large files.
    """
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        yield text[start:end]
        start = end - overlap
        if start < 0:
            start = 0


def parse_info(path: Path) -> Tuple[str, str, str]:
    module_id = path.parent.name       # e.g., module1
    chapter_id = path.stem             # e.g., chapter-1-introduction
    chapter_title = chapter_id.replace("-", " ").replace("_", " ").title()
    return module_id, chapter_id, chapter_title


def collect_markdown_files() -> List[Path]:
    print(f"📂 Looking in: {FRONTEND_DATA_ROOT.resolve()}")   # Debug print
    files = [Path(p) for p in glob.glob(str(FRONTEND_DATA_ROOT / "**" / "chapter*.md"), recursive=True)]
    print(f"🔍 Found {len(files)} markdown files under {FRONTEND_DATA_ROOT.resolve()}")
    for f in files:
        print(f"   ➡️ {f}")
    return files


def main():
    md_paths = collect_markdown_files()
    if not md_paths:
        print("⚠️ No markdown files found. Check FRONTEND_DATA_ROOT path.")
        return

    for md in md_paths:
        print(f"📄 Processing file: {md}")
        module_id, chapter_id, chapter_title = parse_info(md)

        try:
            with open(md, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"⚠️ Error reading {md}: {e}")
            continue

        # Skip huge files
        if len(text) > 10_000_000:  # ~10 MB
            print(f"⚠️ Skipping {md} (too large: {len(text)} chars)")
            continue

        # Create per-file lists to reduce memory usage
        file_chapter_ids = []
        file_section_ids = []
        file_contents = []

        for i, chunk in enumerate(chunk_text(text)):
            file_chapter_ids.append(chapter_id)
            file_section_ids.append(f"{module_id}-{chapter_id}-{i}")
            file_contents.append(chunk)

        if not file_contents:
            print(f"⚠️ No chunks generated for {md}")
            continue

        # Send immediately to Gemini/Qdrant
        try:
            print(f"🚀 Sending {len(file_contents)} chunks from {md} to Gemini...")
            embedding_service.create_embeddings_for_texts(file_chapter_ids, file_section_ids, file_contents)
            print(f"✅ Finished indexing {md}")
        except Exception as e:
            print(f"❌ Embedding failed for {md}: {e}")

    print("🎉 All markdown files processed!")


if __name__ == "__main__":
    main()
