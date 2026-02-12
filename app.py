import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi

st.set_page_config(page_title="StreamMind AI", page_icon="🎬", layout="wide")
st.title("🎬 StreamMind: Chat with your Videos")

with st.sidebar:
    st.header("Credentials")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    st.divider()
    st.header("Content Source")
    video_url = st.text_input("YouTube URL")

if not groq_api_key or not hf_api_key:
    st.warning("Please enter your API Keys.")
    st.stop()

@st.cache_resource
def load_models(groq_key):
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_models(groq_api_key)

def get_video_id(url):
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    elif "be/" in url: return url.split("be/")[1].split("?")[0]
    return None

def process_video(url):
    video_id = get_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL")
        return None
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es'])
        full_text = " ".join([t['text'] for t in transcript_list])
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        
        vs = FAISS.from_texts(chunks, embedding=embeddings)
        return vs.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Transcript Error: {str(e)}")
        st.info("Note: Make sure the video has captions enabled.")
        return None

if video_url:
    if "current_video" not in st.session_state or st.session_state.current_video != video_url:
        with st.spinner("Analyzing video..."):
            retriever = process_video(video_url)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.current_video = video_url
                st.session_state.messages = []
                st.success("Ready to chat!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the video"):
    if "retriever" not in st.session_state:
        st.error("No video data found.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        template = """Answer the question based strictly on the transcript provided.
        Context: {context}
        Question: {input}"""
        
        qa_prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": st.session_state.retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})