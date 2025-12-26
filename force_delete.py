from app.vector_db import vector_db

print('🗑️ Deleting collection...')
try:
    vector_db.client.delete_collection('robotics_book')
    print('✅ Collection deleted!')
except Exception as e:
    print(f'Error: {e}')
