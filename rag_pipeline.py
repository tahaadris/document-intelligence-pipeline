from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(pdf_path: str):
    """
    Loads a PDF, chunks the text, embeds it into a local ChromaDB, 
    and returns a LangChain LCEL retriever chain.
    """
    # 1. Document Loading
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Text Splitting (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Vector Embeddings & Storage
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Prompt Engineering (Strict Grounding)
    template = """Answer the question based STRICTLY on the following context. 
    If the answer is not in the context, say "I do not have this information."
    
    Context: {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. LLM Initialization
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Build LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
