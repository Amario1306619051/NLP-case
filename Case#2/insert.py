from util import get_embeddings
from db_process.database import save_to_database

# Save to database chroma
save_to_database(
    file_path='csv',  # CSV file path
    db_path='path to chromadb file',  # Path where the chromadb database is stored
    collection_name='chromadb collection name',  # Name of the chromadb collection
    LLM_process=False  # Whether to preprocess using LLM
)