import os
import time
import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="C++ AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# PREMIUM DARK LUXURY THEME
# ----------------------------
st.markdown("""
<style>
:root{
  --bg:#070812;
  --bg2:#0f1326;
  --card:rgba(255,255,255,0.06);
  --border:rgba(120,90,255,0.35);
  --accent:#8A5CFF;
  --accent2:#C8A2FF;
  --text:#F1F3FF;
  --muted:#B8C0E6;
  --shadow:0 20px 60px rgba(0,0,0,.6);
}

.stApp{
  background:
    radial-gradient(900px 400px at 10% 10%, rgba(138,92,255,.18), transparent 55%),
    radial-gradient(800px 400px at 90% 10%, rgba(200,162,255,.15), transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color: var(--text);
}

.block-container{
  padding-top: 2rem !important;
  max-width: 900px;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
  font-size: 2.3rem;
  font-weight: 800;
  letter-spacing: 0.3px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  margin-bottom: 0.3rem;
}

.subtitle{
  color: var(--muted);
  margin-bottom: 1.2rem;
}

.hero{
  background: var(--card);
  border:1px solid var(--border);
  border-radius:18px;
  padding:18px;
  box-shadow: var(--shadow);
}

[data-testid="stChatMessage"] > div{
  background: var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:14px;
  box-shadow:0 8px 30px rgba(0,0,0,.4);
}

.stButton>button{
  width:100%;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border:none;
  border-radius:14px;
  padding:.55rem;
  color:white;
  font-weight:600;
}

.stButton>button:hover{
  opacity:0.9;
}

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0a0f20, #121733);
}

section[data-testid="stSidebar"] *{
  color:var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="title">C++ AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions from your C++ Introduction notes.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Local Retrieval Powered Assistant • Fast • Lightweight • Production UI</div>', unsafe_allow_html=True)

load_dotenv()

FILE_PATH = "C++_Introduction.txt"

# ----------------------------
# VECTOR BUILD
# ----------------------------
@st.cache_resource
def build_vector():
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if not os.path.exists(FILE_PATH):
    st.error("C++_Introduction.txt not found in project folder.")
    st.stop()

vectorstore = build_vector()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Top Results", 1, 6, 3)
    retriever.search_kwargs = {"k": k}

    show_sources = st.toggle("Show Sources", value=True)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# CHAT MEMORY
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------
query = st.chat_input("Ask your C++ question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    docs = retriever.invoke(query)

    if docs:
        combined = "\n\n".join([d.page_content for d in docs])
    else:
        combined = "No relevant content found."

    response_placeholder = st.empty()
    full_response = ""

    # Typing animation
    for char in combined:
        full_response += char
        response_placeholder.markdown(full_response)
        time.sleep(0.003)

    if show_sources and docs:
        source_info = "\n\n---\n**Sources:**\n"
        for i, d in enumerate(docs):
            source_info += f"- Chunk {i+1}\n"
        response_placeholder.markdown(full_response + source_info)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
