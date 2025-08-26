import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from postgres import fetch_chat_history

# Set up environment variable (use dotenv in production)
os.environ["GEMINI_API_KEY"] = "AIzaSyBvYKOy0tw7-hWuAL7WTENAVVj4qlPrFxU"
session = None
token = None
history = None

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Gemini Embedding Function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        
        # Handle both single strings and lists of strings
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            input = list(input)
            
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=model, 
                content=text, 
                task_type="retrieval_document", 
                title=title
            )
            embeddings.append(result["embedding"])
        
        return embeddings

# Load ChromaDB collection
def load_chroma_collection(path, name):
    """
    Loads or creates a Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path to the Chroma database.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded or created Chroma Collection.
    """
    try:
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=path)
        
        try:
            collection = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
            print(f"Collection '{name}' loaded successfully.")
        except (chromadb.errors.InvalidCollectionException, ValueError) as e:
            print(f"Collection '{name}' not found or invalid. Creating new collection. Error: {e}")
            collection = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
            print(f"Collection '{name}' created successfully.")
            
        return collection
        
    except Exception as e:
        print(f"Error loading/creating ChromaDB collection: {e}")
        raise e

# Create RAG prompt
def make_rag_prompt(query, session_id, token_id, relevant_passage):
    try:
        history = fetch_chat_history(session_id, token_id)
        print("Raw history:", history)

        # Ensure proper formatting of the conversation history
        formatted_history = "\n".join(
            f"User: {item['message']} | Assistant: {item['response']}" for item in (history or [])
        ) if history else "No prior conversation history available."

        print("Formatted History:\n", formatted_history)

        escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ").strip()

        prompt = f"""
You are a highly intelligent and creative storytelling assistant, designed to enhance the parent-child reading experience and provide engaging, independent storytelling sessions.  

Below is the conversation history from previous interactions between the user and the assistant. Use this history to maintain context, recall relevant details, and ensure consistency in storytelling.  

**Conversation History:**
{formatted_history}

**User's Current Request:**
'{query}'

**Relevant Context or Story Passage:**
'{escaped_passage}'

**Instructions:**
- Carefully analyze the conversation history to understand the ongoing theme, characters, and plot points.
- If the user is continuing a previous story, ensure coherence and logical progression.
- If the user is asking a new question, consider past interactions to provide a relevant and consistent response.
- Maintain a creative, engaging, and child-friendly tone in your storytelling.

**Your Response:**
        """

        print("Generated Prompt:\n", prompt)
        return prompt
        
    except Exception as e:
        print(f"Error creating RAG prompt: {e}")
        # Fallback prompt without history
        return f"""
You are a creative storytelling assistant for children.

User's Request: '{query}'

Context: '{relevant_passage}'

Please provide an engaging, child-friendly response.
        """

# Generate answer using Gemini API
def generate_answer_api(prompt):
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        answer = model.generate_content(prompt)
        return answer.text
        
    except Exception as e:
        print(f"Error generating answer with Gemini API: {e}")
        return f"I'm sorry, I encountered an error while generating a response. Please try again."

# Retrieve relevant passage from ChromaDB
def get_relevant_passage(query, db, n_results):
    try:
        if db.count() == 0:
            print("Warning: ChromaDB collection is empty. No relevant passages found.")
            return ["No relevant context available."]
            
        passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
        return passage if passage else ["No relevant context found."]
        
    except Exception as e:
        print(f"Error retrieving relevant passage: {e}")
        return ["Error retrieving context."]

# Generate answer from retrieved text
def generate_answer(db, query, session_id, token_id):
    try:
        relevant_text = get_relevant_passage(query, db, n_results=3)
        prompt = make_rag_prompt(query, session_id, token_id, relevant_passage="".join(relevant_text))
        answer = generate_answer_api(prompt)
        return answer
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Initialize ChromaDB - moved inside a try-catch block
db = None

def initialize_database():
    global db
    try:
        # Point to the extracted ChromaDB collection
        db_path = os.path.join(BASE_DIR, "chroma_database_edumate")
        db = load_chroma_collection(path=db_path, name="edumate")
        print("Database initialized successfully!")
        print(f"Collection name: {db.name}")
        print(f"Number of items in collection: {db.count()}")
        return True
        
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        print("The application will continue but database functionality will be limited.")
        return False

@app.route("/get-answer", methods=["POST"])
def func():
    try:
        session_id = request.args.get("session_id")
        token_id = request.args.get("token")
        data = request.get_json()
        question = data.get("question", "")
        
        print(f"Session ID: {session_id}, Token ID: {token_id}")
        print(f"Received question: {question}")
        
        if not session_id or not token_id:
            return jsonify({"error": "No session_id or token provided"}), 400
            
        if not question.strip():
            return jsonify({"error": "No question provided"}), 400
            
        if db is None:
            return jsonify({"error": "Database not initialized"}), 500
            
        answer = generate_answer(db, question, session_id, token_id)
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Error occurred in /get-answer: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get-sessionId", methods=["POST"])
def get_session_id():
    global session
    try:
        session_id = request.args.get("session_id")
        token_id = request.args.get("token")
        
        print(f"Received session_id: {session_id}, token: {token_id}")
        
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
            
        session = session_id 
        print(f"Session set to: {session}")
        return jsonify({"sessionId": session})
    
    except Exception as e:
        print(f"Error occurred in /get-sessionId: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "database_initialized": db is not None})

if __name__ == '__main__':
    print("Starting application...")
    initialize_database()
    app.run(debug=True, port=4000)