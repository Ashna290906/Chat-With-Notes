import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("COHERE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"Key length: {len(api_key) if api_key else 0} characters")

if api_key:
    print("Testing direct Cohere import...")
    try:
        import cohere
        print("✅ Cohere module imported successfully")
        co = cohere.Client(api_key)
        print("✅ Cohere client created successfully")
        response = co.embed(
            texts=["Test"],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        print("✅ Cohere API request successful!")
        print(f"Embedding dimensions: {len(response.embeddings[0])}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
