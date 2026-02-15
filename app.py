import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Goodreads AI Librarian", page_icon="📖", layout="wide")
st.title("📖 Goodreads AI Librarian")

st.markdown("""
Upload your Goodreads library CSV to chat about your reading history, find forgotten favorites, or get personalized recommendations!
**To import or export your books, go to goodreads desktop, find My Books, then click on Import and export under Tools on the left.**
""")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get it from console.groq.com")
    hf_api_key = st.text_input("Hugging Face API Key", type="password", help="For embeddings model")
    
    st.divider()
    
    st.header("Upload Library")
    uploaded_file = st.file_uploader("Upload Goodreads CSV", type="csv")
    
    st.divider()
    id_model = "llama-3.3-70b-versatile"
    temperature = 0.5

if not groq_api_key or not hf_api_key:
    st.warning("Please enter both API Keys (Groq and Hugging Face) in the sidebar to begin.")
    st.stop()

@st.cache_resource
def load_models(groq_key, hf_key):
    llm = ChatGroq(api_key=groq_key, model=id_model, temperature=temperature)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

llm, embeddings = load_models(groq_api_key, hf_api_key)

def process_csv_to_retriever(file, embeddings_model):
    """Parse Goodreads CSV and turn each book into a searchable document."""
    df = pd.read_csv(file)
    
    required_cols = ['Title', 'Author']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return None
        
    documents = []
    for _, row in df.iterrows():
        title = str(row.get('Title', 'Unknown'))
        author = str(row.get('Author', 'Unknown'))
        rating = str(row.get('My Rating', 'Unrated'))
        shelves = str(row.get('Bookshelves', ''))
        review = str(row.get('My Review', ''))
        
        content = f"Title: {title}\nAuthor: {author}\nMy Rating: {rating}/5\nShelves/Tags: {shelves}\n"
        if review and review != 'nan':
            content += f"My Review: {review}\n"
            
        doc = Document(page_content=content, metadata={"title": title, "author": author})
        documents.append(doc)
        
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)
    
    return vectorstore.as_retriever(search_kwargs={"k": 10})

if uploaded_file:
    if 'retriever' not in st.session_state:
        with st.spinner("Scanning your shelves and organizing your library..."):
            retriever = process_csv_to_retriever(uploaded_file, embeddings)
            if retriever:
                st.session_state.retriever = retriever
                st.success("Library processed and ready to chat!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("E.g., 'What 5-star fantasy books have I read?' or 'Recommend me a sci-fi book from my to-read shelf'"):
    if not uploaded_file or 'retriever' not in st.session_state:
        st.error("Please upload and process your Goodreads CSV first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        system_prompt = (
            "You are a friendly, knowledgeable AI librarian analyzing a user's Goodreads library. "
            "Use the following retrieved context (which contains books the user has shelved, their ratings, and their reviews) "
            "to answer their question or provide recommendations. "
            "If they ask for recommendations, prioritize books from their own library context. If appropriate, you can suggest outside books that match their taste based on the context. "
            "Keep your responses conversational, insightful, and concise. "
            "If you don't know the answer, just say so.\n\n"
            "Context: {context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = (
            {"context": st.session_state.retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Scanning the shelves..."):
                response = chain.invoke(prompt)

                clean_response = response.split("</think>")[-1].strip() if "</think>" in response else response
                
                st.write(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})