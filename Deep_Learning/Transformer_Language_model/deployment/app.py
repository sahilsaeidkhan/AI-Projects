
import streamlit as st
import model

st.set_page_config(
    page_title="TwinTalk",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom styling
# -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

:root {
    --bg: #101014;
    --panel: #17171f;
    --border: #292936;
    --accent: #a78bfa;
    --text: #f5f3ff;
    --muted: #9694a7;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #17171f;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 27px;
    letter-spacing: -1px;
}

section[data-testid="stSidebar"] p {
    color: var(--muted);
}

/* Header */
.hero {
    text-align: center;
    padding: 35px 0 32px;
}

.hero-icon {
    display: inline-flex;
    width: 58px;
    height: 58px;
    align-items: center;
    justify-content: center;
    border-radius: 19px;
    background: linear-gradient(135deg, #8b5cf6, #c084fc);
    font-size: 27px;
    box-shadow: 0 8px 35px #8b5cf633;
}

.hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 46px;
    font-weight: 700;
    letter-spacing: -2px;
    margin: 17px 0 5px;
    background: linear-gradient(90deg, #fff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 14px;
}

/* Status */
.status {
    display: inline-block;
    color: #86efac;
    background: #163326;
    border: 1px solid #24583c;
    border-radius: 30px;
    padding: 6px 13px;
    font-size: 11px;
    margin-top: 10px;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 17px 20px;
    margin-bottom: 16px;
}

div[data-testid="stChatMessage"] p {
    font-size: 15px;
    line-height: 1.7;
}

/* Input */
div[data-testid="stChatInput"] {
    border: 1px solid #393447;
    border-radius: 17px;
    background: #1c1b26;
}

div[data-testid="stChatInput"] textarea {
    color: white;
    font-size: 15px;
}

/* Buttons */
.stButton button {
    border-radius: 11px;
    border: 1px solid var(--border);
    background: #22212d;
    color: white;
    transition: all 0.2s ease;
}

.stButton button:hover {
    border-color: var(--accent);
    color: #c4b5fd;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 65px 20px;
    border: 1px dashed #343143;
    border-radius: 22px;
    background: #15141d;
    margin-top: 20px;
}

.empty-state h3 {
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    color: #f5f3ff;
}

.empty-state p {
    color: var(--muted);
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #777486;
    font-size: 11px;
    padding: 30px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Session state
# -------------------------

if "conversation" not in st.session_state:
    st.session_state.conversation = []


# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.markdown("# ✦ TwinTalk AI")
    st.caption("Your personal AI conversation companion.")

    st.markdown("---")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

    st.markdown("### About")
    st.caption("A conversational language model built using a custom decoder-only Transformer.")

    st.markdown("---")

    st.markdown("### Model information")
    st.caption("Architecture · Custom Transformer")
    st.caption(f"Runtime · {str(model.DEVICE).upper()}")

    st.markdown("---")
    st.caption("TwinTalk AI · Built with PyTorch")


# -------------------------
# Main header
# -------------------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">✦</div>
    <h1>TwinTalk AI</h1>
    <p>A little conversation can make the day better.</p>
    <span class="status">● MODEL ONLINE</span>
</div>
""", unsafe_allow_html=True)


# -------------------------
# Chat history
# -------------------------

if not st.session_state.conversation:

    st.markdown("""
    <div class="empty-state">
        <h3>Start a conversation</h3>
        <p>Say hello. Let's see where the conversation goes.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    for speaker, message in st.session_state.conversation:
        role = "user" if speaker == "BOY" else "assistant"

        with st.chat_message(
            role,
            avatar="🧑" if role == "user" else "🤖"
        ):
            st.markdown(message)


# -------------------------
# Chat input
# -------------------------

user_message = st.chat_input("Write something...")

if user_message:

    st.session_state.conversation.append(("BOY", user_message))

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_message)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):

            try:
                response = model.generate_girl_response(
                    conversation_history=st.session_state.conversation,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=40
                )

                if not response.strip():
                    response = "Hmm, kya hua? 😅"

            except Exception:
                st.error("Something went wrong while generating a response.")
                st.stop()

            st.markdown(response)

    st.session_state.conversation.append(("GIRL", response))


# -------------------------
# Footer
# -------------------------

st.markdown("""
<div class="footer">
    Powered by a custom Transformer language model · TwinTalk AI
</div>
""", unsafe_allow_html=True)
