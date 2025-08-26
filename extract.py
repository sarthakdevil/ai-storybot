import chromadb
import gdown
import os

def download_and_extract_database():
    """
    Downloads the ChromaDB folder from Google Drive.
    """
    # Google Drive folder ID from the URL
    folder_id = "1fYSWTd_e8M_ZELtrjNPocyV8rDcVxFLr"
    extract_folder = "chroma_database_edumate"
    
    try:
        # Check if the database folder already exists
        if os.path.exists(extract_folder):
            print(f"Database folder '{extract_folder}' already exists. Skipping download.")
            return True
            
        print(f"Downloading ChromaDB folder from Google Drive...")
        
        # Download the entire folder
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}",
            output=extract_folder,
            quiet=False,
            use_cookies=False
        )
        
        print("Download completed!")
        return True
            
    except Exception as e:
        print(f"Error downloading database folder: {e}")
        print("Please try downloading manually from:")
        print(f"https://drive.google.com/drive/folders/{folder_id}")
        return False

def extract_chroma_database_edumate(persist_directory: str = "chroma_database_edumate"):
    """
    Extracts and returns the ChromaDB collection for 'edumate' from the specified directory.
    First ensures the database is downloaded from Google Drive.
    """
    # Ensure database is downloaded
    if not download_and_extract_database():
        raise Exception("Failed to download database")
    
    # Check if the directory exists
    if not os.path.exists(persist_directory):
        raise Exception(f"Database directory '{persist_directory}' not found after download")
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("edumate")
    return collection

# Example usage:
if __name__ == "__main__":
    try:
        collection = extract_chroma_database_edumate()
        print(f"Collection name: {collection.name}")
        print(f"Number of items: {collection.count()}")
    except Exception as e:
        print(f"Error: {e}")