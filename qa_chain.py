from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os

def get_qa_chain(vectorstore, detail_level):
    cohere_key = os.getenv("COHERE_API_KEY")
    model = "command"
    temperature = 0.3
    max_tokens = 2000  # Increased token limit for longer responses

    instruction = ""
    if detail_level == "Detailed":
        instruction = """
        Provide a comprehensive and detailed answer. Include:
        - Key concepts and explanations
        - Relevant examples if applicable
        - Step-by-step reasoning
        - A brief summary at the end
        """
    else:
        instruction = "Provide a concise and clear answer, focusing on the key points."

    prompt = PromptTemplate.from_template(f"""
You are an AI assistant providing expert-level responses. 
Your task is to answer the following question based on the provided context.

### Context:
{{context}}

### Question:
{{question}}

### Instructions:
1. Provide a complete and well-structured answer
2. If the context is insufficient, clearly state what information is missing
3. Use markdown formatting for better readability
4. Include relevant details and explanations
5. Conclude with a brief summary

{instruction}

### Answer:
""")

    # Configure the language model
    llm = Cohere(
        cohere_api_key=cohere_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Configure the retriever to get more context
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
    # Create the QA chain
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
        verbose=True
    )
    
    # Create and return the QA chain
    return RetrievalQA(
        combine_documents_chain=qa_chain,
        retriever=retriever,
        return_source_documents=True,
        input_key="query"
    )
