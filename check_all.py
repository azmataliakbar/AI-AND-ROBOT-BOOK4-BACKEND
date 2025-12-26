from app.vector_db import vector_db

print('Checking ALL vectors for chapter numbers...')

points = vector_db.client.scroll(
    collection_name='robotics_book',
    limit=100,
    with_payload=True
)

# Find vectors WITH chapter numbers
chapters_found = {}
for point in points[0]:
    chapter = point.payload.get('chapter_number')
    if chapter and chapter != '':
        title = point.payload.get('chapter_title', 'Unknown')
        if chapter not in chapters_found:
            chapters_found[chapter] = title

print(f'\nFound {len(chapters_found)} unique chapters:')
for ch in sorted(chapters_found.keys(), key=lambda x: int(x) if x.isdigit() else 0):
    print(f'  Chapter {ch}: {chapters_found[ch][:60]}')
