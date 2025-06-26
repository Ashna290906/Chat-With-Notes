import os
import sys
import requests
from dotenv import load_dotenv

def main():
    print("Minimal Cohere API Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in .env file")
        return
        
    print(f"✅ Found API key (first 10 chars): {api_key[:10]}...")
    
    # Simple GET request to check connectivity
    print("\nTesting connection to Cohere API...")
    try:
        response = requests.get(
            "https://api.cohere.ai/v1/health",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        print(f"✅ Connection successful! Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
