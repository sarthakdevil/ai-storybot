import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import os
import requests
import json
import time
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from postgres import fetch_chat_history

# Set up environment variable (use dotenv in production)
from dotenv import load_dotenv
load_dotenv()

# Stability API Key
STABILITY_KEY = os.getenv("STABILITY_API_KEY")
session = None
token = None
history = None

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Gemini Embedding Function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    
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
        except Exception as e:
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

**IMPORTANT - Response Format:**
You MUST respond with a valid JSON object in one of these two formats:

**For Story Content:**
{{"text": "Your story content here...", "type": "story"}}

**For Interactive Questions with Answer Options:**
{{"questions": [
    {{"question": "Your question here?", "answers": ["Option1", "Option2", "Option3"]}},
    {{"question": "Your second question?", "answers": ["OptionA", "OptionB"]}}
    {{"question": "Your third question?", "answers": ["OptionX", "OptionY", "OptionZ"]}}
], "type": "question"}}

**Instructions:**
- Carefully analyze the conversation history to understand the ongoing theme, characters, and plot points.
- If the user is continuing a previous story, ensure coherence and logical progression.
- If the user is asking a new question, consider past interactions to provide a relevant and consistent response.
- Maintain a creative, engaging, and child-friendly tone in your storytelling.

**When to use each format:**
- Use "story" type when: continuing a narrative, developing plot, telling a complete story segment, or providing descriptive storytelling content.
- Use "question" type when: asked to ask users questions according to the story context.

**Your Response (JSON only):**
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

# Parse JSON response from AI
def parse_ai_json_response(response_text):
    try:
        import json
        
        # Clean up the response text - remove any markdown formatting or extra text
        cleaned_response = response_text.strip()
        
        # Try to find JSON within the response
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = cleaned_response[start_idx:end_idx]
            parsed_json = json.loads(json_str)
            
            # Validate the JSON structure
            if "type" in parsed_json:
                if parsed_json["type"] == "story" and "text" in parsed_json:
                    # Generate image for story content
                    print("=== Generating image for story ===")
                    story_image = generate_story_image(parsed_json["text"])
                    if story_image:
                        parsed_json["image"] = story_image
                        print("Image generated successfully")
                    else:
                        print("Image generation failed or skipped")
                    return parsed_json
                elif parsed_json["type"] == "question" and "questions" in parsed_json:
                    # Ensure questions is a list and validate structure
                    if isinstance(parsed_json["questions"], list):
                        # Validate each question has the required structure
                        valid_questions = []
                        for q in parsed_json["questions"]:
                            if isinstance(q, dict) and "question" in q and "answers" in q:
                                if isinstance(q["answers"], list):
                                    valid_questions.append(q)
                                else:
                                    # Convert answers to list if it's a string
                                    q["answers"] = [q["answers"]] if q["answers"] else []
                                    valid_questions.append(q)
                            elif isinstance(q, str):
                                # Convert old string format to new format
                                valid_questions.append({
                                    "question": q,
                                    "answers": ["Yes", "No", "Maybe"]  # Default answers
                                })
                        
                        parsed_json["questions"] = valid_questions
                        return parsed_json
                    else:
                        # Convert single question to list
                        if isinstance(parsed_json["questions"], dict):
                            parsed_json["questions"] = [parsed_json["questions"]]
                        else:
                            # Convert string to proper format
                            parsed_json["questions"] = [{
                                "question": str(parsed_json["questions"]),
                                "answers": ["Yes", "No", "Maybe"]
                            }]
                        return parsed_json
            
        # If JSON parsing fails, return as story type
        print(f"Could not parse JSON from AI response, defaulting to story type")
        story_response = {
            "text": response_text,
            "type": "story"
        }
        
        # Generate image for fallback story content
        print("=== Generating image for fallback story ===")
        story_image = generate_story_image(response_text)
        if story_image:
            story_response["image"] = story_image
            print("Image generated successfully for fallback")
        
        return story_response
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response_text}")
        # Fallback to story format
        story_response = {
            "text": response_text,
            "type": "story"
        }
        story_image = generate_story_image(response_text)
        if story_image:
            story_response["image"] = story_image
        return story_response
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        story_response = {
            "text": response_text,
            "type": "story"
        }
        story_image = generate_story_image(response_text)
        if story_image:
            story_response["image"] = story_image
        return story_response

# Stable Diffusion functions
def send_generation_request(host, params, files=None):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(host, params, files=None):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        print(f"Polling results at https://api.stability.ai/v2beta/results/{generation_id}")
        response = requests.get(
            f"https://api.stability.ai/v2beta/results/{generation_id}",
            headers={
                **headers,
                "Accept": "*/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response

def generate_story_image(story_text):
    """Generate an image based on story content using Stable Diffusion"""
    try:
        if not STABILITY_KEY:
            print("Stability API key not found, skipping image generation")
            return None
            
        # Create a descriptive prompt from the story
        image_prompt = f"Children's storybook illustration, colorful and friendly style: {story_text[:500]}"
        
        # Parameters for image generation
        params = {
            "prompt": image_prompt,
            "output_format": "png",
            "aspect_ratio": "16:9",
            "style_preset": "fantasy-art"
        }
        
        # Generate image
        host = "https://api.stability.ai/v2beta/stable-image/generate/core"
        response = send_generation_request(host, params)
        
        if response.ok:
            # Convert image to base64 for easy transmission
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return image_base64
        else:
            print(f"Image generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

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

app = Flask(__name__)
# Initialize Flask app
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "https://client-chatbot-two.vercel.app"
]

CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})

# Add a test route to verify the server is working
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "Server is running", "port": os.environ.get("PORT", 4000)})

# Initialize ChromaDB - moved inside a try-catch block
db = None

def initialize_database():
    global db
    try:
        print("=== DATABASE INITIALIZATION ===")
        # Point to the extracted ChromaDB collection
        db_path = os.path.join(BASE_DIR, "chroma_database_edumate")
        print(f"Database path: {db_path}")
        print(f"Path exists: {os.path.exists(db_path)}")
        
        db = load_chroma_collection(path=db_path, name="edumate")
        print("Database initialized successfully!")
        print(f"Collection name: {db.name}")
        print(f"Number of items in collection: {db.count()}")
        return True
        
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        print("The application will continue but database functionality will be limited.")
        db = None
        return False

@app.route("/get-answer", methods=["POST"])
def func():
    try:
        print("=== GET-ANSWER REQUEST RECEIVED ===")
        session_id = request.args.get("session_id")
        token_id = request.args.get("token")
        
        print(f"Raw request args: {request.args}")
        print(f"Session ID: {session_id}, Token ID: {token_id}")
        
        # Get JSON data
        try:
            data = request.get_json()
            print(f"Request JSON data: {data}")
        except Exception as json_error:
            print(f"Error parsing JSON: {json_error}")
            return jsonify({"error": "Invalid JSON data"}), 400
            
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        question = data.get("question", "")
        print(f"Received question: '{question}'")
        
        # Validation
        if not session_id or not token_id:
            print("Missing session_id or token")
            return jsonify({"error": "No session_id or token provided"}), 400
            
        if not question.strip():
            print("Empty question provided")
            return jsonify({"error": "No question provided"}), 400
            
        if db is None:
            print("Database not initialized")
            return jsonify({"error": "Database not initialized"}), 500
            
        print("=== Starting AI response generation ===")
        # Generate the AI response
        answer = generate_answer(db, question, session_id, token_id)
        print(f"AI raw response: {answer}")
        
        # Parse the JSON response from AI
        response_data = parse_ai_json_response(answer)
        print(f"Parsed response: {response_data}")
        
        print("=== Sending successful response ===")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"ERROR occurred in /get-answer: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
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
    import os
    print("Starting application...")
    initialize_database()
    port = int(os.environ.get("PORT", 4000))  # Default to port 4000
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)  # Enable debug for better error messages



