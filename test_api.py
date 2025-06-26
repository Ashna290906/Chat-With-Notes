import os
import sys
import requests
from dotenv import load_dotenv

def main():
    print("Cohere API Connection Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in .env file")
        return
        
    print(f"✅ Found API key (first 10 chars): {api_key[:10]}...")
    
    # Test connection to Cohere API
    url = "https://api.cohere.ai/v1/health"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("\nTesting connection to Cohere API...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"✅ Connection successful! Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
    
    # Test embeddings endpoint
    print("\nTesting embeddings endpoint...")
    url = "https://api.cohere.ai/v1/embed"
    data = {
        "texts": ["Test connection"],
        "model": "embed-english-v3.0",
        "input_type": "search_document"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        print("✅ Embeddings test successful!")
        print(f"Response keys: {list(response.json().keys())}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Embeddings test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    main()
