from app.vector_db import vector_db

print('Checking specific chapters...')

points = vector_db.client.scroll(
    collection_name='robotics_book',
    limit=10,
    with_payload=True
)

print(f'\nFound {len(points[0])} sample vectors:\n')
for i, point in enumerate(points[0][:10], 1):
    meta = point.payload
    chapter = meta.get('chapter_number', 'N/A')
    title = meta.get('chapter_title', 'N/A')
    if title != 'N/A':
        title = title[:50]
    module = meta.get('module', 'N/A')
    print(f'{i}. Chapter: {chapter} | Title: {title}... | Module: {module}')
