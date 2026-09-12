"""Simple UI. Run: streamlit run src/app.py"""

import requests
import streamlit as st

from config import API_TIMEOUT_SECONDS, BACKEND_URL, USER_ACCESS


def call_backend(endpoint, payload):
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}", json=payload,
            timeout=(5, API_TIMEOUT_SECONDS),
        )
    except requests.Timeout:
        raise RuntimeError("The request timed out. Reset the conversation before retrying; it may still be processing.") from None
    except requests.ConnectionError:
        raise RuntimeError("Cannot connect to the backend. Start FastAPI and try again.") from None
    except requests.RequestException:
        raise RuntimeError("The backend request failed. Please try again.") from None
    if not response.ok:
        try:
            detail = response.json().get("detail", "Request failed.")
        except ValueError:
            detail = "The backend returned an error. Check its terminal."
        if not isinstance(detail, str):
            detail = "Invalid request. Check your input and try again."
        raise RuntimeError(detail)
    try:
        return response.json()
    except ValueError:
        raise RuntimeError("The backend returned an unreadable response.") from None


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Cited sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.caption(f"[{source.get('citation_number', index)}] {source['company'].upper()} · {source['source']} · PDF page {source['page']}")
                    st.text(source["text"])


st.set_page_config(page_title="Secure Chat Interface", page_icon="📄")
st.title("Secure Chat Interface")
st.caption("Ask questions about the earnings documents.")
st.session_state.setdefault("messages", [])

if not st.session_state.get("email"):
    with st.form("login"):
        selected_user = st.selectbox("Select user", list(USER_ACCESS))
        if st.form_submit_button("Login"):
            try:
                with st.spinner("Loading your conversation..."):
                    result = call_backend("/history", {
                        "email": selected_user, "session_id": "default",
                    })
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.session_state.email = selected_user
                st.session_state.session_id = "default"
                st.session_state.messages = result["messages"]
                st.rerun()
    st.stop()

email = st.session_state.email
session_id = st.session_state.session_id
with st.sidebar:
    st.write(f"Logged in as: {email}")
    st.write("Allowed companies: " + ", ".join(company.upper() for company in USER_ACCESS[email]))
    if st.button("Reset Conversation"):
        try:
            with st.spinner("Resetting conversation..."):
                call_backend("/reset", {"email": email, "session_id": session_id})
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.session_state.messages = []
            st.session_state.session_id = "default"
            st.rerun()
    if st.button("Logout"):
        # Clear the UI; keep backend history for the next login.
        st.session_state.pop("email", None)
        st.session_state.pop("session_id", None)
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    show_message(message)

if question := st.chat_input("Ask about a company, metric, and reporting period", max_chars=4000):
    if question.strip():
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        show_message(user_message)
        try:
            with st.spinner("Searching documents and preparing an answer..."):
                result = call_backend("/chat", {
                    "email": email, "session_id": session_id, "question": question,
                })
        except RuntimeError as error:
            st.session_state.messages.append({
                "role": "assistant", "content": str(error), "sources": [],
            })
        else:
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"], "sources": result["sources"],
            })
        st.rerun()
