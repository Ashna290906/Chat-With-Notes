from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

class CustomCohereEmbeddings(Embeddings):
    def __init__(self, cohere_api_key):
        self.client = cohere.Client(cohere_api_key)

    def embed_documents(self, texts):
        return self.client.embed(texts=texts, model="embed-english-v3.0", input_type="search_document").embeddings

    def embed_query(self, text):
        return self.client.embed(texts=[text], model="embed-english-v3.0", input_type="search_query").embeddings[0]

def build_vectorstore(chunks):
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise ValueError("COHERE_API_KEY is missing.")
    embeddings = CustomCohereEmbeddings(key)
    return FAISS.from_texts(chunks, embedding=embeddings)
