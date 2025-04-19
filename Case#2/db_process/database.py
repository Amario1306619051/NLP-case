import pandas as pd
# Import dari llama_index
# Import dari modul lain
import chromadb


from util import get_embeddings
from LLM.model import humanize, define_paragraph


def split_into_batches(data_list: list[str], batch_size: int = 5) -> list[list[str]]:
    """
    Split a list into smaller batches of a given size.

    Args:
        data_list (list[str]): The list to be split.
        batch_size (int): The size of each batch.

    Returns:
        list[list[str]]: A list of batches, where each batch is a list of strings.
    """
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def save_to_database(file_path:str, db_path:str, collection_name:str, LLM_process:bool = False):
    """
    Save data from a CSV file into a ChromaDB database.

    Args:
        file_path (str): Path to the xlsx file containing the dataset.
        db_path (str): Path to the ChromaDB database.
        collection_name (str): Name of the collection in the database.

    Returns:
        None
    """
    print(f"Loading dataset from {file_path}...")
    dataset = pd.read_excel(file_path)
    Questions = split_into_batches(dataset['Question'].tolist())
    metadatas = split_into_batches(dataset['Answer'].tolist())

    print(f"Connecting to the database at {db_path}...")
    # Load ChromaDB index
    database = chromadb.PersistentClient(path=db_path)
    collection = database.get_or_create_collection(collection_name)

    # Retrieve the last ID if the function is called multiple times
    last_id = 0
    try:
        existing_documents = collection.get()
        if existing_documents and 'ids' in existing_documents:
            last_id = max(map(int, existing_documents['ids']))
    except Exception as e:
        print(f"Unable to retrieve existing IDs. Starting from 0. Error: {e}")

    print(f"Processing and inserting data into the collection '{collection_name}'...")
    for i, (Questions_batch, metadatas_batch) in enumerate(zip(Questions, metadatas)):
        # Generate unique IDs for the batch
        batch_ids = [str(last_id + j + 1) for j in range(len(Questions_batch))]
        last_id += len(Questions_batch)

        if LLM_process:
            Questions_batch = [define_paragraph(q, a) for q, a in zip(Questions_batch, metadatas_batch)]
        else:
            Questions_batch = [f"{question} { answer}" for question, answer in zip(Questions_batch, metadatas_batch)]
        print(Questions_batch)
        embeddings = get_embeddings(Questions_batch)
        metadatas_batch = [{'answer': metadata} for metadata in metadatas_batch]

        # Insert into ChromaDB
        collection.upsert(
            documents=Questions_batch,
            metadatas=metadatas_batch,
            embeddings=embeddings.tolist(),
            ids=batch_ids
        )
        print(f"Batch {i} inserted successfully.")

    print("All data has been successfully inserted into the database.")


def query_database(db_path:str, collection_name:str, query:str):
    """
    Query the ChromaDB database for a specific question.

    Args:
        db_path (str): Path to the ChromaDB database.
        collection_name (str): Name of the collection in the database.
        query (str): The question to be queried.

    Returns:
        list: A list of results from the database.
    """
    print(f"Connecting to the database at {db_path}...")
    # Load ChromaDB index
    database = chromadb.PersistentClient(path=db_path)
    collection = database.get_collection(collection_name)

    print(f"Querying the collection '{collection_name}'...")
    results = collection.query(
        query_embeddings=get_embeddings([query]).tolist(),
        n_results=2
    )

    return results

def check_data(db_path, collection_name):
    """
    Check the data in the ChromaDB database.

    Args:
        db_path (str): Path to the ChromaDB database.
        collection_name (str): Name of the collection in the database.

    Returns:
        None
    """
    print(f"Connecting to the database at {db_path}...")
    # Load ChromaDB index
    database = chromadb.PersistentClient(path=db_path)
    collection = database.get_collection(collection_name)

    print(f"Checking data in the collection '{collection_name}'...")
    results = collection.get()

    print(results)