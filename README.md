# Document Intelligence Q&A Pipeline

A Retrieval-Augmented Generation (RAG) system built to query proprietary enterprise documents (PDFs) with grounded, deterministic outputs to eliminate LLM hallucination.

## Architecture & Tech Stack
* **Framework:** LangChain (LCEL)
* **User Interface:** Streamlit
* **Vector Store:** ChromaDB (Local)
* **Embeddings & LLM:** OpenAI API (Swappable for enterprise models)
* **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

## Core Features
1. **Semantic Text Chunking:** Implements recursive character splitting (1000 tokens with 200 overlap) to preserve semantic meaning across page boundaries.
2. **Dense Vector Retrieval:** Uses cosine similarity within ChromaDB to fetch the top 3 most relevant text chunks based on the user's query.
3. **Strict Context Grounding:** Custom prompt templates force the model to answer *only* using the retrieved context, returning "I do not have this information" if the answer is absent from the source document.
4. **Interactive Dashboard:** Features a Streamlit UI for seamless document uploading and querying.

## Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Export your API key: `export OPENAI_API_KEY="your-key-here"`
4. Launch the interface: `streamlit run app.py`
