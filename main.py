import google.generativeai as genai
import os
import requests
import json
import time
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
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

# Create storytelling prompt
def make_rag_prompt(query, session_id, token_id):
    try:
        history = fetch_chat_history(session_id, token_id)
        print("Raw history:", history)

        # Ensure proper formatting of the conversation history
        formatted_history = "\n".join(
            f"User: {item['message']} | Assistant: {item['response']}" for item in (history or [])
        ) if history else "No prior conversation history available."

        print("Formatted History:\n", formatted_history)

        prompt = f"""
You are a highly intelligent and creative storytelling assistant, designed to enhance the parent-child reading experience and provide engaging, independent storytelling sessions.  

Below is the conversation history from previous interactions between the user and the assistant. Use this history to maintain context, recall relevant details, and ensure consistency in storytelling.  

**Conversation History:**
{formatted_history}

**User's Current Request:**
'{query}'

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
- Use "question" type when: the user requests to "let me choose next", "ask me questions", or when you want to engage the user with interactive choices about the story direction.

**Your Response (JSON only):**
        """

        print("Generated Prompt:\n", prompt)
        return prompt
        
    except Exception as e:
        print(f"Error creating storytelling prompt: {e}")
        # Fallback prompt without history
        return f"""
You are a creative storytelling assistant for children.

User's Request: '{query}'

Please provide an engaging, child-friendly response in JSON format:
- For stories: {{"text": "Your story here", "type": "story"}}
- For questions: {{"questions": [{{"question": "Your question?", "answers": ["Option1", "Option2"]}}], "type": "question"}}
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

# Generate answer using Gemini API
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

# Generate answer from user query
def generate_answer(query, session_id, token_id):
    try:
        prompt = make_rag_prompt(query, session_id, token_id)
        answer = generate_answer_api(prompt)
        return answer
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

app = Flask(__name__)

# Enable CORS for all origins
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
     allow_headers=["Content-Type", "Authorization"], supports_credentials=True)

# Add a test route to verify the server is working
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "Server is running", "port": os.environ.get("PORT", 4000)})

@app.route("/get-answer", methods=["POST", "OPTIONS"])
@cross_origin(origins="*", supports_credentials=True, methods=["GET", "POST", "OPTIONS"])
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
            
        print("=== Starting AI response generation ===")
        # Generate the AI response
        answer = generate_answer(question, session_id, token_id)
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

@app.route("/get-sessionId", methods=["POST", "OPTIONS"])
@cross_origin(origins="*", supports_credentials=True, methods=["GET", "POST", "OPTIONS"])
def get_session_id():
    global session
    try:
        # Accept session_id and token from query params or JSON body.
        session_id = request.args.get("session_id") or None
        # account for clients that may send ?token or ??token
        token_id = request.args.get("token") or request.args.get("?token") or None

        # If not present in query params, try JSON body (use silent=True so OPTIONS preflight doesn't raise)
        data = None
        try:
            data = request.get_json(silent=True)
        except Exception:
            data = None

        if not session_id and data:
            session_id = data.get("session_id") or data.get("sessionId")
        if not token_id and data:
            token_id = data.get("token") or data.get("auth_token") or data.get("token_id")

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
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    import os
    print("Starting application...")
    port = int(os.environ.get("PORT", 4000))  # Default to port 4000
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)  # Enable debug for better error messages



