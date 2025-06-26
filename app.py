import streamlit as st
import time
from utils import extract_text, split_text
from vectorstore import build_vectorstore
from qa_chain import get_qa_chain
from datetime import datetime, timedelta
from functools import wraps
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_message(prompt):
    """Process a user message and generate a response."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        # Initialize response text
        response_text = ""
        
        # Check API key first
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            st.error("COHERE_API_KEY not found. Please set it in your environment variables or .env file.")
            response_text = "Please set up your COHERE_API_KEY first."
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return None
        
        # Check if QA chain is initialized
        if not st.session_state.qa_chain:
            st.warning("Please upload a document first to enable chat.")
            response_text = "Please upload a document first to enable chat."
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return None
            
        # Create a placeholder for the response
        response_placeholder = st.empty()
        
        # Show processing message with emoji
        processing_emojis = ["🔍 Searching...", "🧠 Thinking...", "📚 Analyzing...", "💡 Generating..."]
        processing_index = 0
        
        def update_processing_message():
            nonlocal processing_index
            emoji = processing_emojis[processing_index % len(processing_emojis)]
            response_placeholder.markdown(f"{emoji}")
            processing_index += 1
            return emoji
            
        # Initial processing message
        update_processing_message()
        
        # Test connection first
        test_url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {cohere_key}",
            "Content-Type": "application/json",
            "Request-Source": "python-sdk"
        }
        
        try:
            # Update processing message before making the request
            update_processing_message()
            
            # Make the test request
            response = requests.post(
                test_url,
                headers=headers,
                json={
                    "prompt": "Test connection",
                    "model": "command-light",
                    "max_tokens": 10
                },
                timeout=5
            )
            
            # Debug: Print response details
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Headers: {response.headers}")
            print(f"API Response Body: {response.text}")
            
            response.raise_for_status()
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            st.error(f"API connection error: {error_msg}")
            response_text = f"API Error: {error_msg}. Please check your API key and try again."
            response_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Network Error: {str(e)}"
            st.error(f"API connection error: {error_msg}")
            response_text = f"Network Error: {error_msg}. Please check your internet connection."
            response_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return None
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            st.error(f"API connection error: {error_msg}")
            response_text = f"Unexpected Error: {error_msg}. Please try again."
            response_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            return None
        
        # Check if this is a summarization request
        is_summary = any(word in prompt.lower() for word in ["summarize", "summary", "key points", "main points"])
        
        # For summarization, modify the prompt to be more specific
        query_text = f"Provide a concise summary of the key points from the document. {prompt}" if is_summary else prompt
        
        # Get the response with timeout
        try:
            # Show processing message
            update_processing_message()
            
            # Get the response from QA chain with timeout
            with st.spinner(''):  # Empty spinner to handle interruption
                # Start a timer to update the processing message
                start_time = time.time()
                last_update = 0
                
                def check_update():
                    nonlocal last_update
                    current_time = time.time()
                    if current_time - last_update > 2:  # Update every 2 seconds
                        update_processing_message()
                        last_update = current_time
                
                # Run the QA chain with timeout
                try:
                    response = st.session_state.qa_chain({"query": query_text})
                    
                    # Debug: Print the raw response
                    print(f"Raw response: {response}")
                    
                    # Handle the response
                    if isinstance(response, dict):
                        response_text = response.get("result", "")
                        
                        # If no result but we have source documents, indicate that
                        if not response_text and response.get("source_documents"):
                            response_text = "I found some information in the document, but couldn't generate a specific answer. Could you try rephrasing your question?"
                    else:
                        response_text = str(response) if response else ""
                    
                    # Clean up the response text
                    response_text = response_text.strip()
                    
                    # Handle empty or very short responses
                    if not response_text or len(response_text) < 5:
                        response_text = "The document doesn't contain specific information about that. Could you try asking in a different way?"
                    
                    # Update the response in the UI
                    response_placeholder.markdown(response_text)
                    
                except TimeoutError as e:
                    raise TimeoutError("The request took too long") from e
                except Exception as e:
                    raise e
                
        except TimeoutError as e:
            error_msg = f"The request took too long: {str(e)}"
            st.error(error_msg)
            response_text = "I'm sorry, the request timed out. Please try a more specific question."
            response_placeholder.markdown(response_text)
            print(error_msg)
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            print(error_msg)
            if "context" in str(e).lower():
                response_text = "The document doesn't contain information about that. Could you try asking in a different way?"
            else:
                response_text = "I'm having trouble with that request. Could you try rephrasing your question?"
            response_placeholder.markdown(response_text)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Save to chat history
        if st.session_state.current_chat:
            st.session_state.chat_history[st.session_state.current_chat] = st.session_state.messages.copy()
        
        # Rerun to update the chat display
        st.rerun()

def timeout(seconds=30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            error = None
            
            def target():
                nonlocal result, error
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    error = e
            
            import threading
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
            if error:
                raise error
            return result
        return wrapper
    return decorator

# Page config
st.set_page_config(
    page_title="Chat with Your Notes",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main layout */
    html, body, #root, .stApp {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 100vh !important;
    }
    .stApp {
        background: linear-gradient(135deg, #6A1B9A 0%, #1976D2 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientBG 15s ease infinite !important;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Chat container background */
    .stChatContainer {
        background: linear-gradient(135deg, #6A1B9A 0%, #1976D2 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientBG 15s ease infinite !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Message bubbles */
    .stChatMessage[data-testid="user"] .stChatMessageContent {
        background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    .stChatMessage[data-testid="assistant"] .stChatMessageContent {
        background: linear-gradient(135deg, #1976D2 0%, #2196F3 100%) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Floating particles */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    
    .particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 50%;
        pointer-events: none;
        animation: float 15s infinite linear;
    }
    
    @keyframes float {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-1000px) rotate(720deg);
            opacity: 0;
        }
    }
    
    /* Typing animation */
    .typing-animation {
        border-right: 2px solid #fff;
        animation: blink 0.7s step-end infinite;
        display: inline-block;
    }
    
    @keyframes blink {
        from, to { border-color: transparent; }
        50% { border-color: #fff; }
    }
    .main {
        background: transparent !important;
        min-height: 100vh;
        display: flex;
        color: #000000 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #333333 !important;
        padding: 1.5rem 1rem !important;
        color: #ffffff !important;
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        max-width: 800px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 1.5rem;
        border-radius: 16px !important;
        border: 2px solid #e1bee7 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
        border: 2px solid #7E57C2 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #5E35B1 !important;
        box-shadow: 0 0 0 3px rgba(126, 87, 194, 0.3) !important;
        outline: none !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        background: linear-gradient(45deg, #7E57C2 0%, #2196F3 100%) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2) !important;
        background: linear-gradient(45deg, #6A1B9A 0%, #1976D2 100%) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 12px 0 !important;
        margin: 0 0 16px 0 !important;
        border: none !important;
        background: transparent !important;
    }
    
    .stChatMessage > div:first-child {
        align-items: flex-start !important;
    }
    
    .stChatMessage .stChatMessageContent {
        padding: 14px 18px !important;
        border-radius: 18px !important;
        max-width: 80% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
    }
    
    .stChatMessage[data-testid="user"] .stChatMessageContent {
        background: linear-gradient(135deg, #D1C4E9 0%, #B39DDB 100%) !important;
        color: #000000 !important;
        margin-left: auto !important;
        border-bottom-right-radius: 4px !important;
        border: 1px solid #9C27B0 !important;
        font-weight: 500;
    }
    
    .stChatMessage[data-testid="assistant"] .stChatMessageContent {
        background: linear-gradient(135deg, #BBDEFB 0%, #90CAF9 100%) !important;
        color: #000000 !important;
        margin-right: auto !important;
        border-bottom-left-radius: 4px !important;
        border: 1px solid #2196F3 !important;
        font-weight: 500;
    }
    
    .stChatMessage .stChatMessageAvatar {
        width: 36px !important;
        height: 36px !important;
        font-size: 18px !important;
        background: linear-gradient(45deg, #7E57C2 0%, #5E35B1 100%) !important;
        color: white !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        margin-right: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stChatMessage[data-testid="assistant"] .stChatMessageAvatar {
        background: linear-gradient(45deg, #2196F3 0%, #1976D2 100%) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #ec407a 0%, #9c27b0 100%) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #e91e63 0%, #7b1fa2 100%) !important;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    
    .chat-history-item {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9375rem;
    }
    
    .chat-history-item:hover {
        background-color: #f3f4f6;
    }
    
    .chat-history-item.active {
        background-color: #eef2ff;
        color: #4f46e5;
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .stFileUploader:hover {
        border-color: #9ca3af;
    }
    
    /* Chat header */
    .chat-header {
        padding: 1.5rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .chat-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Login / Signup UI
def show_auth_ui():
    st.markdown("""<style>
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .gradient-text {
            background: linear-gradient(90deg, #9c27b0, #e91e63, #ff9800, #9c27b0);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 5s ease infinite, fadeIn 0.8s ease-out;
            font-weight: 700;
            text-align: center;
            margin: 0 0 2rem 0;
            font-size: 2.2rem;
            letter-spacing: -0.5px;
        }
        /* Remove default margins and padding */
        html, body, #root, .stApp {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 100vh !important;
            overflow: auto !important;
        }
        .stApp { 
            background: linear-gradient(135deg, #0f0c29, #4a1b7a, #302b63, #1a1a2e) !important;
            background-size: 300% 300% !important;
            animation: gradient 15s ease infinite !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 100vh !important;
            padding: 20px !important;
        }
        .auth-container {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(15px) saturate(1.5);
            border-radius: 16px;
            padding: 2.5rem !important;
            width: 100%;
            max-width: 450px !important;
            animation: fadeIn 0.6s ease-out;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > div > input {
            border-radius: 12px !important; 
            padding: 14px 16px !important;
            background: rgba(30, 30, 30, 0.8) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.3s ease !important;
            font-size: 15px;
        }
        .stTextInput > div > div > input:focus, 
        .stTextInput > div > div > input:hover {
            border-color: #9c27b0 !important;
            box-shadow: 0 0 0 2px rgba(156, 39, 176, 0.3) !important;
            background: rgba(40, 40, 40, 0.9) !important;
        }
        .stButton > button {
            background: linear-gradient(45deg, #9c27b0, #e91e63) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 14px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 15px;
            margin-top: 10px;
            box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(156, 39, 176, 0.4) !important;
            background: linear-gradient(45deg, #8e24aa, #d81b60) !important;
        }
        .stTabs {
            margin-bottom: 1.5rem;
        }
        .stTabs [role="tablist"] {
            gap: 8px;
            margin-bottom: 1.5rem;
        }
        .stTabs [role="tab"] {
            color: #888 !important;
            background: transparent !important;
            border: none !important;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(156, 39, 176, 0.2) !important;
            color: #ffffff !important;
            border: 1px solid rgba(156, 39, 176, 0.4) !important;
            box-shadow: 0 4px 12px rgba(156, 39, 176, 0.2);
        }
        .stTabs [role="tab"]:hover {
            color: #ccc !important;
        }
        .stAlert {
            border-radius: 12px !important;
            animation: slideIn 0.4s ease-out;
        }
        .stAlert [data-testid="stMarkdownContainer"] {
            color: #fff !important;
        }
    </style>""", unsafe_allow_html=True)

    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='gradient-text'>Welcome to Chat with Notes</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if email and password:
                st.session_state.user = email
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Please fill all fields")
    with tab2:
        email = st.text_input("New Email", key="signup_email")
        password = st.text_input("Create Password", type="password", key="signup_pwd")
        confirm = st.text_input("Confirm Password", type="password", key="signup_cpwd")
        if st.button("Create Account"):
            if not email or not password or not confirm:
                st.error("All fields required")
            elif password != confirm:
                st.error("Passwords do not match")
            else:
                st.session_state.user = email
                st.success("Account created!")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if not st.session_state.user:
    show_auth_ui()

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Sidebar
with st.sidebar:
    # New Chat Button
    if st.button("New Chat", use_container_width=True, type="primary"):
        # Only create new chat if there are messages and it's not a duplicate
        if st.session_state.messages:
            # Create a unique chat ID based on timestamp
            chat_id = f"Chat_{int(time.time())}"
            # Check if this chat already exists in history
            if chat_id not in st.session_state.chat_history:
                st.session_state.chat_history[chat_id] = st.session_state.messages.copy()
                st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.last_uploaded = None
    
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Drag and drop a file here or click to browse",
        type=["pdf", "txt", "docx", "pptx", "ppt", "xlsx", "xls"],
        label_visibility="collapsed"
    )
    
    # Sample Questions Section
    if st.session_state.last_uploaded:
        st.markdown("### Sample Questions")
        st.markdown("""
        <div style="background: #1e1e1e; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="color: #a0a0a0; margin-bottom: 0.5rem;">Try asking about:</p>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="color: #ffffff; margin: 0.5rem 0;">
                    <span style="color: #6366f1;">•</span> What is the main topic of this document?
                </li>
                <li style="color: #ffffff; margin: 0.5rem 0;">
                    <span style="color: #6366f1;">•</span> Can you summarize the key points?
                </li>
                <li style="color: #ffffff; margin: 0.5rem 0;">
                    <span style="color: #6366f1;">•</span> What are the main arguments presented?
                </li>
                <li style="color: #ffffff; margin: 0.5rem 0;">
                    <span style="color: #6366f1;">•</span> What evidence is provided?
                </li>
                <li style="color: #ffffff; margin: 0.5rem 0;">
                    <span style="color: #6366f1;">•</span> Can you explain [specific concept]?
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded'):
            # Clear previous state
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            
            # Get file extension and map to filetype
            file_extension = uploaded_file.name.split('.')[-1].lower()
            filetype_mapping = {
                'pdf': 'pdf',
                'docx': 'docx',
                'doc': 'docx',
                'pptx': 'pptx',
                'ppt': 'pptx',  # Handle both .ppt and .pptx
                'xlsx': 'xlsx',   # Excel files
                'xls': 'xls',     # Older Excel format
                'txt': 'txt'
            }
            
            if file_extension not in filetype_mapping:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            try:
                text = extract_text(uploaded_file, filetype_mapping[file_extension])
                docs = split_text(text)
                st.session_state.vectorstore = build_vectorstore(docs)
                st.session_state.qa_chain = get_qa_chain(
                    vectorstore=st.session_state.vectorstore,
                    detail_level='Normal'
                )
                st.session_state.last_uploaded = uploaded_file
                st.session_state.messages = []
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                
                # Display the chat interface with the file name
                st.markdown("""
                    <div class="chat-header" style="background: #1e1e1e; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;">
                        <h2 style="color: #ffffff; margin: 0; font-size: 1.25rem;">
                            Chat with """ + uploaded_file.name + """
                        </h2>
                        <p style="color: #a0a0a0; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
                            Ask questions about the document above
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
    
    st.markdown("---")
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for chat_id in st.session_state.chat_history:
            if st.button(chat_id, key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.messages = st.session_state.chat_history[chat_id].copy()
                st.session_state.current_chat = chat_id
                st.rerun()
    
    st.markdown("---")
    
    # User info and logout
    st.markdown(f"""
        <div style="margin-top: auto; padding: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #e5e7eb; 
                            display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 14px; font-weight: 600; color: #4b5563;">
                        {st.session_state.user[0].upper()}
                    </span>
                </div>
                <div>
                    <div style="font-weight: 500; color: #ffffff;">{st.session_state.user}</div>
                    <div style="font-size: 0.75rem; color: #ffffff;">Free Plan</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Add floating particles container
st.markdown('<div class="particles" id="particles-js"></div>', unsafe_allow_html=True)

# Add JavaScript for particles
st.markdown("""
<script>
// Create particles
function createParticles() {
    const container = document.getElementById('particles-js');
    if (!container) return;
    
    // Clear existing particles
    container.innerHTML = '';
    
    // Create 50 particles
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random size between 2px and 6px
        const size = Math.random() * 4 + 2;
        
        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        
        // Random animation duration between 10s and 20s
        const duration = Math.random() * 10 + 10;
        
        // Apply styles
        Object.assign(particle.style, {
            width: `${size}px`,
            height: `${size}px`,
            left: `${posX}%`,
            top: `${posY}%`,
            animationDuration: `${duration}s`,
            animationDelay: `-${Math.random() * 15}s`
        });
        
        container.appendChild(particle);
    }
}

// Initialize particles when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createParticles);
} else {
    createParticles();
}

// Recreate particles when the page changes (for Streamlit's SPA behavior)
const observer = new MutationObserver((mutations) => {
    if (document.getElementById('particles-js') && 
        !document.querySelector('.particle')) {
        createParticles();
    }
});

observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Main Chat Area
if st.session_state.last_uploaded:
    # If a document is loaded, show the file name in the header
    st.markdown(f"""
        <div class="chat-header" style="background: rgba(0, 0, 0, 0.7); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; backdrop-filter: blur(10px);">
            <h2 style="color: white; margin: 0;">Chat with {st.session_state.last_uploaded.name}</h2>
        </div>
    """, unsafe_allow_html=True)
else:
    # Show animated welcome message when no document is loaded
    st.markdown("""
        <div style="text-align: center; margin: 5rem 0; padding: 2rem;">
            <h1 style="color: white; font-size: 2.5rem; margin-bottom: 1rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                <span id="welcome-text"></span><span class="typing-animation"></span>
            </h1>
            <p style="color: #e0e0e0; font-size: 1.2rem; margin-top: 1.5rem; opacity: 0; text-shadow: 0 1px 2px rgba(0,0,0,0.3);" id="subtext">Upload a document to start chatting with your notes</p>
        </div>""", unsafe_allow_html=True)
    
    # Add the JavaScript separately to avoid syntax issues
    st.markdown("""
    <script>
        // Typewriter effect
        const text = "Welcome to Chat with Notes";
        let i = 0;
        const speed = 100;
        
        function typeWriter() {
            if (i < text.length) {
                document.getElementById("welcome-text").innerHTML += text.charAt(i);
                i++;
                setTimeout(typeWriter, speed);
                
                if (i === text.length) {
                    setTimeout(function() {
                        const subtext = document.getElementById("subtext");
                        subtext.style.transition = "opacity 1s ease-in-out";
                        subtext.style.opacity = "1";
                    }, 500);
                }
            }
        }
        
        // Start the typing effect
        if (document.readyState === 'complete') {
            setTimeout(typeWriter, 1000);
        } else {
            window.addEventListener('load', function() {
                setTimeout(typeWriter, 1000);
            });
        }
    </script>
    """, unsafe_allow_html=True)
    
# Chat input with unique key
if prompt := st.chat_input("Type your message here...", key=f"chat_input_{st.session_state.current_chat or 'new'}"):
    process_message(prompt)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

# Empty state - only show when there are no messages
if not st.session_state.messages:
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; 
                justify-content: center; height: 60vh; text-align: center; padding: 2rem;">
        <h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">
            Welcome to Chat with Notes
        </h2>
        <p style="color: #6b7280; max-width: 500px; margin-bottom: 2rem;">
            Upload a document to get started or select a previous chat from the sidebar.
        </p>
        <button class="stButton" style="margin: 0 auto;">
            <div data-testid="stMarkdownContainer" class="stMarkdown" style="width: 100%;">
                <p>Upload Document</p>
            </div>
        </button>
    </div>
    """, unsafe_allow_html=True)


