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
                # Iterate over the response chunks as they arrive
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        # Forward the raw chunk directly to the client
                        yield chunk
            except Exception as e:
                print(f"Error during streaming from provider: {e}")
            finally:
                response.close()

        # The content type for Server-Sent Events (SSE) is text/event-stream
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

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
