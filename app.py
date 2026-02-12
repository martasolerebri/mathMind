import streamlit as st
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="MathMind Pro", page_icon="🧮", layout="wide")
st.title("🧮 MathMind Pro: RAG Math Tutor")

with st.sidebar:
    st.header("Credentials")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    st.divider()
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Math PDF", type="pdf")
    if st.button("Clear Database"):
        if "math_retriever" in st.session_state:
            del st.session_state.math_retriever
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()

if not groq_api_key or not hf_api_key:
    st.warning("Please enter your API Keys to start.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key):
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key)

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file.getbuffer())
        file_path = tf.name
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

if uploaded_file and "math_retriever" not in st.session_state:
    with st.spinner("Indexing PDF..."):
        retriever = process_pdf(uploaded_file)
        if retriever:
            st.session_state.math_retriever = retriever
            st.success("PDF ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the math in your PDF"):
    if "math_retriever" not in st.session_state:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        template = """You are an expert Math Tutor. Use the provided context to answer.
        Always use LaTeX for mathematical notation ($ for inline, $$ for block).
        Context: {context}
        Question: {input}
        Answer:"""
        
        qa_prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": st.session_state.math_retriever, "input": RunnablePassthrough()}
            | qa_prompt | llm | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})