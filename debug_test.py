import os
import sys
import time
import traceback
from dotenv import load_dotenv
import cohere

def log_message(message):
    """Log message to both console and file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open('cohere_debug.log', 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

def test_cohere_connection():
    log_message("Starting Cohere API test...")
    
    try:
        # Load environment variables
        load_dotenv()
        log_message("1. Environment variables loaded")
        
        # Get API key
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            log_message("❌ Error: COHERE_API_KEY not found in .env file")
            return False
        
        log_message(f"✅ Found COHERE_API_KEY (first 8 chars): {api_key[:8]}...")
        
        # Test direct API call
        log_message("2. Testing direct API call...")
        import requests
        
        url = "https://api.cohere.ai/v1/health"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            log_message(f"3. Health check status: {response.status_code}")
            log_message(f"Response: {response.text}")
        except Exception as e:
            log_message(f"❌ Health check failed: {str(e)}")
            log_message(f"Error type: {type(e).__name__}")
            return False
        
        # Test Cohere client
        log_message("4. Testing Cohere client...")
        try:
            co = cohere.Client(api_key)
            log_message("✅ Cohere client created successfully")
            
            # Test embedding
            log_message("5. Testing embeddings...")
            response = co.embed(
                texts=["Test connection to Cohere API"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            
            if response.embeddings and len(response.embeddings) > 0:
                log_message(f"✅ Success! Embedding dimensions: {len(response.embeddings[0])}")
                return True
            else:
                log_message("❌ Received empty embeddings in the response")
                return False
                
        except cohere.CohereAPIError as e:
            log_message(f"❌ Cohere API Error: {str(e)}")
            log_message(f"Status code: {getattr(e, 'status_code', 'N/A')}")
            return False
            
    except Exception as e:
        log_message(f"❌ Unexpected error: {str(e)}")
        log_message(f"Error type: {type(e).__name__}")
        log_message("\n" + traceback.format_exc())
        return False
    finally:
        log_message("Test completed.")
        log_message("=" * 50 + "\n")

if __name__ == "__main__":
    test_cohere_connection()
