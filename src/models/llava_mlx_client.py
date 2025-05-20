#!/usr/bin/env python3
"""
LLaVA-MLX Client
A simple client for interacting with LLaVA-1.6-Vicuna 13B running on MLX via LM Studio
"""

import argparse
import base64
import json
import os
import requests
import time
from PIL import Image
import io

# Default configurations
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "llava-v1.6-vicuna-13b"

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoded string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def query_llava(image_path, prompt, api_url=DEFAULT_API_URL, model=DEFAULT_MODEL, max_tokens=1024, temperature=0.1):
    """Send a query to the LLaVA model via LM Studio API"""
    
    # Encode the image to base64
    image_base64 = None
    if image_path: # Only encode if an image_path is provided
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return "Error: Failed to encode image."
    
    # Prepare the payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful image analysis assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Adjust payload for text-only messages if no image_base64
    if not image_base64:
        payload["messages"][1]["content"] = prompt # Simpler content for text-only
    
    headers = {"Content-Type": "application/json"}
    
    # Debug information
    print(f"Sending request to {api_url}")
    print(f"Using model: {model}")
    print(f"Prompt: {prompt}")
    
    try:
        # Send the request
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        elapsed_time = time.time() - start_time
        
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content']
            print(f"Request completed in {elapsed_time:.2f} seconds")
            return answer
        else:
            print(f"Warning: Unexpected response format: {result}")
            return "Error: Unexpected response format"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: API request failed - {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f". Details: {e.response.text}"
        print(error_msg)
        return error_msg

def resize_image_if_needed(image_path, max_size=1024):
    """Resize the image if it's too large, returns path to the resized image"""
    try:
        img = Image.open(image_path)
        
        # Check if resizing is needed
        width, height = img.size
        if width <= max_size and height <= max_size:
            return image_path  # No resizing needed
            
        # Calculate new dimensions while preserving aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create filename for the resized image
        filename, ext = os.path.splitext(image_path)
        resized_path = f"{filename}_resized{ext}"
        
        # Save the resized image
        resized_img.save(resized_path)
        print(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        return resized_path
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_path  # Return original path on error

def check_server_status(api_url=DEFAULT_API_URL):
    """Check if the LM Studio server is running"""
    try:
        # Send a simple request to check server status
        response = requests.get(api_url.replace("/v1/chat/completions", "/v1/models"))
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Query LLaVA model via LM Studio")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="LM Studio API URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens in response")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for text generation")
    
    args = parser.parse_args()
    
    # Check if the server is running
    if not check_server_status(args.api_url):
        print(f"Error: Cannot connect to LM Studio at {args.api_url}")
        print("Make sure LM Studio is running and the model is loaded.")
        return
    
    # Resize image if needed
    image_path = resize_image_if_needed(args.image)
    
    # Query the model
    response = query_llava(
        image_path=image_path,
        prompt=args.prompt,
        api_url=args.api_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\n--- Model Response ---")
    print(response)
    
if __name__ == "__main__":
    main() 