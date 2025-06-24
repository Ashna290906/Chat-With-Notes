import streamlit as st

# In-memory user storage for demo purposes
USERS = {}

# --- Custom CSS for glassmorphism login/signup UI ---
st.markdown("""
<style>
body {
    margin: 0;
    padding: 0;
}
.main-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    color: white;
}
.auth-box {
    background: rgba(255, 255, 255, 0.05);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    width: 100%;
    max-width: 400px;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid #444 !important;
    color: white !important;
    border-radius: 10px;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    padding: 0.75rem;
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #5a67d8, #6b46c1);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# --- Authentication logic ---
def login():
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if email in USERS and USERS[email] == password:
            st.session_state.user = email
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

def signup():
    st.subheader("📝 Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

    if st.button("Create Account"):
        if not email or not password or not confirm:
            st.error("⚠️ All fields are required")
        elif password != confirm:
            st.error("⚠️ Passwords do not match")
        elif email in USERS:
            st.error("⚠️ Email already registered")
        else:
            USERS[email] = password
            st.success("✅ Account created! Please log in.")

# --- Main logic ---
def main():
    st.markdown("<div class='main-container'><div class='auth-box'>", unsafe_allow_html=True)

    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        st.success(f"👋 Welcome, **{st.session_state.user}**!")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
    else:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login()
        with tab2:
            signup()

    st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()   