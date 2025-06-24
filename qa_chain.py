from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os

def get_qa_chain(vectorstore, detail_level):
    cohere_key = os.getenv("COHERE_API_KEY")
    model = "command"
    temperature = 0.3
    max_tokens = 1000  # Balanced token limit for quality and speed
    num_generations = 1  # Single generation for faster response
    max_retries = 2  # Limit retries to prevent timeouts

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

    prompt = PromptTemplate.from_template(f"""
# DETAILED EXPERT ANALYSIS

## SOURCE CONTEXT:
{{context}}

## USER'S QUERY:
{{question}}

## RESPONSE PARAMETERS:
- **Response Length**: 500-800 words
- **Detail Level**: Focused and relevant depth
- **Structure**: Follow the outline below
- **Evidence**: Include key examples and data
- **Formatting**: Use markdown for clarity

## INSTRUCTIONS FOR AI:
1. **Exhaustive Coverage**: Leave no stone unturned in your analysis
2. **Multiple Perspectives**: Consider all possible angles and viewpoints
3. **Depth of Detail**: Go multiple levels deep in your explanations
4. **Evidence-Based**: Support every claim with concrete evidence
5. **Structured Flow**: Maintain logical progression between sections
6. **Visual Thinking**: Include detailed descriptions of visual elements
7. **Critical Analysis**: Evaluate strengths, weaknesses, and limitations
8. **Practical Applications**: Provide real-world implementation details

## REQUIRED RESPONSE STRUCTURE:
{instruction}

## ADDITIONAL REQUIREMENTS:
- If any information is missing from the context, explicitly state what's needed
- Use markdown extensively for maximum readability
- Include hypothetical examples where real ones aren't available
- Cross-reference different parts of the analysis
- Use analogies to explain complex concepts
- Address potential counter-arguments
- Provide historical context where relevant
- Include relevant formulas, calculations, or technical details

## FINAL OUTPUT INSTRUCTIONS:
1. Begin with a comprehensive executive summary
2. Follow the detailed structure below exactly
3. Maintain academic rigor while remaining accessible
4. Use clear section headers and subheaders
5. Include numerous examples and case studies
6. Conclude with actionable insights and next steps
7. Provide extensive references and resources

## BEGIN YOUR ULTRA-DETAILED RESPONSE:
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
