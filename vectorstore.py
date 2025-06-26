import os
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import cohere
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

class DirectCohereEmbeddings(Embeddings):
    """Custom embeddings class that uses Cohere client directly"""
    
    def __init__(self, model: str = "embed-english-v3.0"):
        self.model = model
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Cohere"""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return [list(embedding) for embedding in response.embeddings]
        except Exception as e:
            print(f"Error in embed_documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Cohere"""
        return self.embed_documents([text])[0]

def build_vectorstore(chunks: List[str]) -> FAISS:
    """
    Build a FAISS vector store from text chunks using Cohere embeddings.
    
    Args:
        chunks: List of text chunks to be embedded and stored
        
    Returns:
        FAISS: A FAISS vector store containing the embedded chunks
    """
    try:
        print("Initializing Cohere embeddings...")
        
        # Initialize our custom embeddings
        embeddings = DirectCohereEmbeddings()
        
        print(f"Creating FAISS vector store with {len(chunks)} chunks...")
        
        # Process in smaller batches to avoid timeouts
        batch_size = 10
        vectorstore = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            if vectorstore is None:
                # First batch - create new vector store
                vectorstore = FAISS.from_texts(
                    texts=batch,
                    embedding=embeddings,
                    metadatas=[{"source": f"chunk-{j}"} for j in range(i, min(i + batch_size, len(chunks)))]
                )
                print(f"Created new vector store with batch {batch_num}")
            else:
                # Subsequent batches - add to existing vector store
                vectorstore.add_texts(
                    texts=batch,
                    metadatas=[{"source": f"chunk-{j}"} for j in range(i, min(i + batch_size, len(chunks)))]
                )
                print(f"Added batch {batch_num} to vector store")
        
        if vectorstore is None:
            raise ValueError("No documents were processed - vector store is empty")
            
        print("Successfully created vector store with all chunks")
        return vectorstore
        
    except Exception as e:
        print(f"Error in build_vectorstore: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise
