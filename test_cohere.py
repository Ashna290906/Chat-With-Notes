import os
from dotenv import load_dotenv
import cohere

def test_cohere_connection():
    print("Starting Cohere API test...")
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in .env file")
        return False
    
    print("✅ Found COHERE_API_KEY in .env")
    print(f"API Key starts with: {api_key[:8]}...")
    
    try:
        print("\n1. Creating Cohere client...")
        co = cohere.Client(api_key)
        
        print("2. Sending test request to Cohere API...")
        response = co.embed(
            texts=["This is a test of the Cohere API connection. " + 
                  "If you can read this, the connection is working properly."],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        print("\n3. Response received!")
        
        if response.embeddings and len(response.embeddings) > 0:
            print(f"✅ Success! Embedding dimensions: {len(response.embeddings[0])}")
            print(f"First 10 dimensions of the embedding: {response.embeddings[0][:10]}")
            return True
        else:
            print("❌ Received empty embeddings in the response")
            return False
            
    except cohere.CohereAPIError as e:
        print(f"\n❌ Cohere API Error: {str(e)}")
        print(f"Status code: {getattr(e, 'status_code', 'N/A')}")
        print(f"Headers: {getattr(e, 'headers', 'N/A')}")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\nTest completed.")

if __name__ == "__main__":
    test_cohere_connection()
