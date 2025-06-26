import sys
import os
from dotenv import load_dotenv

def main():
    # Redirect stdout and stderr to a file
    with open('debug_output.txt', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        
        print("Starting debug output...")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        # Check if .env file exists
        env_path = os.path.join(os.getcwd(), '.env')
        print(f"Checking for .env file at: {env_path}")
        print(f".env exists: {os.path.exists(env_path)}")
        
        # Load environment variables
        print("Loading environment variables...")
        load_dotenv()
        
        # Check if API key is loaded
        api_key = os.getenv("COHERE_API_KEY")
        print(f"API Key loaded: {'Yes' if api_key else 'No'}")
        if api_key:
            print(f"API Key starts with: {api_key[:8]}...")
            print(f"API Key length: {len(api_key)} characters")
        
        # Test basic Python functionality
        try:
            import cohere
            print("✅ Cohere module imported successfully")
            print(f"Cohere version: {cohere.__version__}")
            
            # Test creating a client
            try:
                client = cohere.Client(api_key)
                print("✅ Cohere client created successfully")
                
                # Test a simple API call
                try:
                    response = client.embed(
                        texts=["Test"],
                        model="embed-english-v3.0",
                        input_type="search_document"
                    )
                    print("✅ Embedding request successful!")
                    print(f"Response type: {type(response)}")
                    print(f"Embedding dimensions: {len(response.embeddings[0])}")
                except Exception as e:
                    print(f"❌ Embedding request failed: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    
            except Exception as e:
                print(f"❌ Client creation failed: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                
        except ImportError as e:
            print(f"❌ Failed to import cohere: {str(e)}")
            print("Try running: pip install cohere")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
        
        print("\nDebug information written to debug_output.txt")

if __name__ == "__main__":
    main()
