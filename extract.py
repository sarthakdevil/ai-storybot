import chromadb

def extract_chroma_database_edumate(persist_directory: str = "chroma_database_edumate"):
    """
    Extracts and returns the ChromaDB collection for 'edumate' from the specified directory.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("edumate")
    return collection

# Example usage:
if __name__ == "__main__":
    collection = extract_chroma_database_edumate()
    print(f"Collection name: {collection.name}")
    print(f"Number of items: {collection.count()}")