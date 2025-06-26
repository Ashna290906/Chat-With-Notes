import os
import sys
import requests
from dotenv import load_dotenv

def main():
    print("Testing Cohere API connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in .env file")
        return False
    
    print("✅ Found COHERE_API_KEY in .env")
    print(f"Key length: {len(api_key)} characters")
    
    # Test direct HTTP request
    url = "https://api.cohere.ai/v1/embed"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Request-Source": "python-sdk"
    }
    
    data = {
        "texts": ["Test connection"],
        "model": "embed-english-v3.0",
        "input_type": "search_document"
    }
    
    print("\nSending test request to Cohere API...")
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=10  # 10 second timeout
        )
        response.raise_for_status()
        print(f"✅ Success! Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return False

if __name__ == "__main__":
    main()
