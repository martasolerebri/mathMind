import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="MathMind AI", page_icon="🧮", layout="wide")
st.title("🧮 MathMind: Your AI Math Tutor")
st.markdown("Solve complex problems and learn the logic behind them using **Groq** and **LaTeX**.")

with st.sidebar:
    st.header("Credentials")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    st.divider()
    st.info("This tutor uses RAG to fetch mathematical rules and Groq to solve equations in milliseconds.")

if not groq_api_key or not hf_api_key:
    st.warning("Please enter your API Keys to start.")
    st.stop()

@st.cache_resource
def load_math_engine(groq_key):
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Simple RAG: Basic math rules for the AI to follow
    math_rules = [
        "Pythagorean theorem: a^2 + b^2 = c^2",
        "Quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a",
        "Derivative of x^n is n*x^(n-1)",
        "Integral of 1/x is ln|x|",
        "Area of a circle: πr^2",
        "Chain rule: d/dx [f(g(x))] = f'(g(x)) * g'(x)"
    ]
    vectorstore = FAISS.from_texts(math_rules, embedding=embeddings)
    return llm, vectorstore.as_retriever()

llm, retriever = load_math_engine(groq_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Enter a math problem (e.g., 'Solve 2x + 5 = 15' or 'Derive sin(x)^2')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    template = """You are a professional Math Tutor. 
    1. Use the following math rules as reference: {context}
    2. Solve the user's problem step-by-step.
    3. Use LaTeX for ALL mathematical formulas. Wrap equations in $$ for blocks and $ for inline.
    4. Be concise and clear.

    Problem: {input}
    Solution:"""
    
    qa_prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | qa_prompt | llm | StrOutputParser()
    )

    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})