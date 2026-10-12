import streamlit as st
import os
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="Document Intelligence Q&A", layout="centered")
st.title("📄 Enterprise Document Intelligence")
st.write("Upload a proprietary PDF to securely query its contents using Retrieval-Augmented Generation.")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Save file temporarily for the PyPDFLoader
    temp_pdf_path = "temp_document.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded successfully! Building vector index...")

    # Initialize the RAG chain and store it in session state
    if "rag_chain" not in st.session_state:
        with st.spinner("Chunking text and generating embeddings..."):
            st.session_state.rag_chain = build_rag_chain(temp_pdf_path)

    st.divider()
    
    # Query Interface
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Retrieving relevant context and generating answer..."):
            # Invoke the LangChain pipeline
            answer = st.session_state.rag_chain.invoke(question)
            
            st.markdown("### Answer")
            st.info(answer)
