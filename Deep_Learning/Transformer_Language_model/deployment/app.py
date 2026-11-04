
import streamlit as st
import model

st.set_page_config(
    page_title="TwinTalk AI",
    page_icon="💬",
    layout="centered"
)

st.title("TwinTalk AI")
st.caption("A conversational AI powered by our custom Transformer.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

with st.sidebar:
    st.title("TwinTalk AI")
    st.caption("Your AI conversation companion.")

    if st.button("Clear conversation"):
        st.session_state.conversation = []
        st.rerun()

    st.divider()
    st.caption("Model: Custom Transformer")
    st.caption(f"Device: {model.DEVICE}")

for speaker, message in st.session_state.conversation:
    role = "user" if speaker == "BOY" else "assistant"
    with st.chat_message(role):
        st.markdown(message)

user_message = st.chat_input("Type your message...")

if user_message:
    st.session_state.conversation.append(("BOY", user_message))

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.generate_girl_response(
                conversation_history=st.session_state.conversation,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )

            if not response.strip():
                response = "Hmm, kya? 😅"

            st.markdown(response)

    st.session_state.conversation.append(("GIRL", response))
