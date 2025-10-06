import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Define Pydantic models for request body validation
class GenerationRequest(BaseModel):
    modelName: str
    messages: List[Dict[str, Any]]

# Load provider data from the JSON file
def load_providers():
    # Assuming provider.json is in the parent directory of the backend folder
    provider_file_path = os.path.join(os.path.dirname(__file__), 'provider.json')
    try:
        with open(provider_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError("provider.json not found. Make sure it's in the project root.")
    except json.JSONDecodeError:
        raise RuntimeError("Error decoding provider.json.")

model_providers = load_providers()

def get_model_details(model_name: str):
    for provider in model_providers:
        for model in provider['models']:
            if model['name'] == model_name:
                return {
                    "apiName": model['apiName'],
                    "url": provider['url'],
                    "apiKey": provider.get('apiKey', '')
                }
    return None

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate")
async def generate(request_data: GenerationRequest):
    model_details = get_model_details(request_data.modelName)

    if not model_details:
        raise HTTPException(status_code=404, detail=f"Model '{request_data.modelName}' not found in provider.json")

    api_url = model_details['url']
    api_key = model_details['apiKey']
    api_name = model_details['apiName']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        'model': api_name,
        'messages': request_data.messages,
        'stream': True
    }

    try:
        # Use requests with stream=True to make the external API call
        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=300)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        def stream_generator():
            try:
                # Use iter_lines to process SSE data correctly
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # For SSE, the actual data is prefixed with "data: "
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[len('data: '):]
                            if json_str.strip() == '[DONE]':
                                continue # OpenAI-compatible signal for end of stream
                            try:
                                # Parse the JSON to access the content
                                data = json.loads(json_str)
                                choices = data.get('choices', [])
                                if choices and 'delta' in choices[0] and 'content' in choices[0]['delta']:
                                    content_chunk = choices[0]['delta']['content']
                                    if content_chunk:
                                        # We send back the raw text content to the frontend
                                        yield content_chunk.encode('utf-8')
                            except json.JSONDecodeError:
                                # If it's not JSON, it might be a malformed line or something else.
                                # We can choose to ignore or log it.
                                # For our purpose, we'll try to pass it on if it's not a control message.
                                print(f"Could not decode JSON from line: {json_str}")
                                pass # Or yield decoded_line.encode('utf-8') if needed
                        else:
                             # If a line doesn't start with 'data: ', it might be a comment or empty line in the SSE stream
                             pass
            except Exception as e:
                print(f"Error during streaming from provider: {e}")
            finally:
                response.close()

        # We will stream back raw text chunks, not JSON
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except requests.exceptions.RequestException as e:
        print(f"Error calling external API: {e}")
        error_detail = f"Failed to connect to the AI provider at {api_url}. Error: {e}"
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except json.JSONDecodeError:
                error_detail = e.response.text
        raise HTTPException(status_code=502, detail=error_detail)


@app.get("/")
def read_root():
    return {"message": "Netsim AI Backend is running with FastAPI."}

if __name__ == "__main__":
    import uvicorn
    # This block allows running the script directly for development `python main.py`
    uvicorn.run(app, host="127.0.0.1", port=5000)
