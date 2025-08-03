import requests

def fetch_chat_history(session_id, token, page=1, limit=3):
    """
    Fetch chat history from the API
    """
    url = f"http://localhost:5000/chat/latest/{session_id}?page={page}&limit={limit}"
    headers = {
        "Authorization": f"Bearer {token}",  # Add Bearer token
        "Content-Type": "application/json"  # Ensure proper content type
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        data = response.json()
        
        # Ensure `data` is a dictionary
        if isinstance(data, list):
            print("Unexpected list response:", data)
            return None  # Or handle list response differently if needed
        
        # Extract messages and responses safely
        history = [
            {"message": item.get("message", ""), "response": item.get("response", "")}
            for item in data.get("data", []) if isinstance(item, dict)  # Ensure `item` is a dictionary
        ]
        
        return history

    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None
