import streamlit as st
from utils import extract_text, split_text
from vectorstore import build_vectorstore
from qa_chain import get_qa_chain
from datetime import datetime

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
        background: #000000 !important;
    }
    .main {
        background: transparent !important;
        min-height: 100vh;
        display: flex;
        color: #ffffff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #121212 !important;
        border-right: 1px solid #333333 !important;
        padding: 1.5rem 1rem !important;
        color: #ffffff !important;
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        max-width: 800px;
        margin: 0 auto;
        background: #121212 !important;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333333 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        background: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        background-color: #4f46e5 !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #4338ca !important;
        transform: translateY(-1px) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 12px 0 !important;
        margin: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        max-width: 80%;
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-message-author-role="user"] {
        margin-left: auto !important;
    }
    
    /* Hide the Streamlit header and footer */
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c7d2fe;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a5b4fc;
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
        if st.session_state.messages:  # Only create new chat if there are messages
            chat_id = f"Chat {len(st.session_state.chat_history) + 1}"
            st.session_state.chat_history[chat_id] = st.session_state.messages.copy()
            st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.last_uploaded = None
    
    st.markdown("---")
    
    # Document Upload Section
    # Detail Level Selector
    detail_level = st.radio(
        "Response Detail Level",
        ("Concise", "Detailed"),
        horizontal=True,
        index=0,
        key="detail_level"
    )
    
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Drag and drop a file here or click to browse",
        type=["pdf", "txt", "docx", "pptx", "ppt", "xlsx", "xls"],
        label_visibility="collapsed"
    )
    
    # Process uploaded file
    if uploaded_file is not None and uploaded_file != st.session_state.get('last_uploaded'):
        with st.spinner("Processing document..."):
            try:
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
                filetype = filetype_mapping.get(file_extension)
                
                # Reset file pointer to beginning for reading
                uploaded_file.seek(0)
                
                if not filetype:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
                text = extract_text(uploaded_file, filetype)
                docs = split_text(text)
                st.session_state.vectorstore = build_vectorstore(docs)
                st.session_state.qa_chain = get_qa_chain(
                    vectorstore=st.session_state.vectorstore,
                    detail_level=st.session_state.detail_level
                )
                st.session_state.last_uploaded = uploaded_file
                st.session_state.messages = []
                st.success("Document processed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
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

# Main Chat Area
st.markdown("""
    <div class="chat-header">
        <h1 class="chat-title">Chat with Notes</h1>
    </div>
""", unsafe_allow_html=True)

# Display chat messages
if not st.session_state.messages:
    # Empty state
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
else:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message Chat with Notes..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        if st.session_state.qa_chain:
            with st.spinner("Generating response..."):
                try:
                    # Call the QA chain with the correct input key 'query'
                    response = st.session_state.qa_chain({"query": prompt})
                    # The response contains 'result' key with the answer
                    response_text = response.get("result", "I couldn't generate a response.")
                    st.markdown(response_text)
                except Exception as e:
                    st.error("Sorry, I encountered an error. Please try again.")
                    response_text = "I'm sorry, I couldn't process your request."
        else:
            st.warning("Please upload a document first to enable chat.")
            response_text = "Please upload a document first to enable chat."
    
    # Add assistant response to messages
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Save to chat history
    if st.session_state.current_chat:
        st.session_state.chat_history[st.session_state.current_chat] = st.session_state.messages
    
    st.rerun()
