from app.vector_db import vector_db

print('Checking ALL 279 vectors...')

# Get ALL vectors (scroll with offset)
all_points = []
offset = None

while True:
    result = vector_db.client.scroll(
        collection_name='robotics_book',
        limit=100,
        offset=offset,
        with_payload=True
    )
    points, next_offset = result
    all_points.extend(points)
    
    if next_offset is None:
        break
    offset = next_offset

print(f'Total vectors retrieved: {len(all_points)}')

# Find all unique chapters
chapters_found = {}
for point in all_points:
    chapter = point.payload.get('chapter_number')
    if chapter and chapter != '':
        title = point.payload.get('chapter_title', 'Unknown')
        module = point.payload.get('module', 'Unknown')
        if chapter not in chapters_found:
            chapters_found[chapter] = (title, module)

print(f'\nFound {len(chapters_found)} unique chapters:\n')
for ch in sorted(chapters_found.keys(), key=lambda x: int(x) if x.isdigit() else 0):
    title, module = chapters_found[ch]
    print(f'Chapter {ch}: {title[:50]}')

# Check for missing chapters
expected = [str(i).zfill(2) for i in range(1, 38)]
missing = [ch for ch in expected if ch not in chapters_found]
if missing:
    print(f'\n❌ Missing chapters: {missing}')
else:
    print('\n✅ All chapters 01-37 present!')
