from django.shortcuts import render
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the API key from environment variables

load_dotenv()
genai_api_key = os.environ.get('GEMINI_API_KEY')

if not genai_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


llm = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)
# Create your views here.
# Function to manage prompt size
def manage_prompt_size(prompt, context, max_tokens=1000000, chunk_size=900000):
    tokens = llm.count_tokens(prompt)
    base_tokens = int(tokens.total_tokens)

    # Calculate the remaining tokens available for context
    available_tokens = max_tokens - base_tokens
    
    # Split the context into chunks that fit within the available tokens
    prompts = []
    current_chunk = ""
    current_chunk_tokens = 0
    print(f"Base tokens: {base_tokens}")
    
    for piece in context:
        piece_tokens = llm.count_tokens(piece).total_tokens
        print(f"Available tokens: {available_tokens}, Current chunk tokens: {current_chunk_tokens}, Peice tokens: {piece_tokens}")
        
        if current_chunk_tokens + piece_tokens <= available_tokens:
            current_chunk += piece + "\n"
            current_chunk_tokens += piece_tokens
        else:
            prompts.append(prompt.replace("context_here", current_chunk))
            current_chunk = piece + "\n"
            current_chunk_tokens = piece_tokens
    
    # Add the last chunk
    if current_chunk:
        prompts.append(prompt.replace("context_here", current_chunk))
    
    return prompts