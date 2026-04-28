import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("rag-doc-chat")
st.caption("Upload a document and ask questions about it.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "pdf", "docx", "md"]
    )
    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            st.error("Upload failed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response = requests.post(
            f"{API_URL}/ask/stream",
            json={"question": question, "session_id": "streamlit"},
            stream=True
        )
        answer = st.write_stream(
            chunk.decode("utf-8") for chunk in response.iter_content(chunk_size=None)
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})