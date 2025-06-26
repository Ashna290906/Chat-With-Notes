from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os

def get_qa_chain(vectorstore, detail_level):
    cohere_key = os.getenv("COHERE_API_KEY")
    model = "command"  # Using the more capable model for better responses
    temperature = 0.7  # Higher temperature for more creative and detailed responses
    max_tokens = 1000  # Increased token limit for more comprehensive answers
    num_generations = 1
    max_retries = 1
    
    # Add prompt suffix to encourage detailed responses
    prompt_suffix = ("""\nIMPORTANT: When creating tables, use EXACTLY this markdown format:
    
    | Column 1 | Column 2 | Column 3 |
    |----------|----------|----------|
    | Data 1   | Data 2   | Data 3   |
    | Data 4   | Data 5   | Data 6   |
    
    RULES FOR TABLES:
    1. Must start and end with blank lines
    2. Must use pipe (|) for column separation
    3. Must include header separator row with ---
    4. No HTML tags allowed
    5. For lists within tables, separate items with semicolons
    
    For regular text:
    - Use markdown bullet points (- or *)
    - Use **bold** and *italic* for emphasis
    - Never use HTML tags
    - Ensure proper spacing around headings and paragraphs""")
    
    # Add repetition penalty to reduce redundancy
    repetition_penalty = 1.2

    instruction = ""
    if detail_level == "Detailed":
        instruction = """
        # DETAILED RESPONSE REQUIREMENTS
        
        ## CORE REQUIREMENTS:
        - Concise yet comprehensive response (500-800 words)
        - Focus on key aspects with relevant details
        - 2-3 levels of nested details
        - Balanced analysis from key perspectives
        
        ## MANDATORY SECTIONS (expand each with maximum detail):
        
        1. **COMPREHENSIVE INTRODUCTION** 
           - Detailed background and context
           - Historical development and evolution
           - Current state of the art
           - Relevance and importance
        
        2. **DEEP DIVE INTO KEY CONCEPTS**
           - Detailed definitions and explanations
           - Theoretical foundations
           - Technical specifications (if applicable)
           - Relationships between concepts
           - Visual representations (describe in detail if needed)
        
        3. **MICROSCOPIC ANALYSIS**
           - Break down into smallest possible components
           - Detailed examination of each component
           - Interactions between components
           - Potential variations and edge cases
        
        4. **EVIDENCE & EXAMPLES**
           - Multiple real-world examples (5+)
           - Case studies with detailed analysis
           - Statistical data and research findings
           - Expert opinions and quotes
           - Visual evidence (describe in detail)
        
        5. **COMPARATIVE ANALYSIS**
           - Compare with similar/related concepts
           - Advantages and disadvantages
           - Use cases for different scenarios
           - Performance metrics and benchmarks
        
        6. **PRACTICAL IMPLEMENTATION**
           - Step-by-step implementation guide
           - Common pitfalls and how to avoid them
           - Best practices and pro tips
           - Tools and resources needed
        
        7. **CRITICAL EVALUATION**
           - Strengths and weaknesses
           - Limitations and constraints
           - Current challenges
           - Future developments and trends
        
        8. **COMPREHENSIVE CONCLUSION**
           - Summary of key points
           - Final analysis and synthesis
           - Actionable insights
           - Future outlook
        
        9. **EXTENDED RESOURCES**
           - Academic papers and studies
           - Books and publications
           - Online courses and tutorials
           - Tools and software
           - Expert communities and forums
        10.**important**
           - Answer should be complete
           - answer should always be complete never leave incomplete answer
           - answer should be to the point in 2-4 paragraph unless asked in detail or in depth
           - answer asked in detsil or depth it should be in detail with 7-8 level of depth
           - answer should be in simple language
           - if asked in tabular form or difference form then answer in tabular form 
           - if asked in bullet points or points then answer in bullet points or point form
           - if asked in list form then answer in list form
           - answer should be concise to the point unless asked in detail or in depth
        
        ## FORMATTING REQUIREMENTS:
        - Use Markdown extensively (headings, subheadings, lists, tables, etc.)
        - Include visual structure with dividers and spacing
        - Use bold/italic for emphasis
        - Create clear visual hierarchy
        - Use blockquotes for important notes
        
        ## WRITING STYLE:
        - Academic yet accessible
        - Precise and unambiguous
        - Thorough but well-organized
        - Engaging and informative
        - Professional but not overly formal
        """
    else:
        instruction = "Provide a concise and clear answer, focusing on the key points."

    # Create the base prompt
    base_prompt = """
You are an expert assistant providing detailed, comprehensive answers based on the following context. 

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a thorough, well-structured answer in 3-5 paragraphs
2. Include relevant examples and explanations
3. Use markdown formatting (headings, lists, bold/italic) for better readability
4. If the question is about a concept, include:
   - Definition
   - Key characteristics
   - Real-world applications
   - Related concepts
5. For comparison questions, use tables or bullet points
6. If unsure, say so and explain what information would be needed

DETAILED ANSWER:"""
    
    # Combine with the prompt suffix
    full_prompt = base_prompt + prompt_suffix
    
    # Create the prompt template
    prompt = PromptTemplate.from_template(full_prompt)
 
    # Configure the language model
    llm = Cohere(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cohere_api_key=cohere_key
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