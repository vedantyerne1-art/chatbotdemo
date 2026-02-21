import streamlit as st
import os
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
    page_title=" C++ Chatbot",
    page_icon="👑",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# ROYAL THEME (CSS)
# ----------------------------
st.markdown(
    """
<style>
/* --- Royal palette --- */
:root{
  --bg1:#05060A;
  --bg2:#0B0E18;
  --card:#0F1426;
  --card2:#101A33;
  --gold:#D4AF37;
  --gold2:#F5D97A;
  --text:#EAF0FF;
  --muted:#AAB4D6;
  --border:rgba(212,175,55,.25);
  --shadow: 0 18px 60px rgba(0,0,0,.55);
}

/* Whole app background */
.stApp{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(212,175,55,.18), transparent 55%),
              radial-gradient(900px 500px at 80% 20%, rgba(130,90,255,.18), transparent 60%),
              linear-gradient(180deg, var(--bg1), var(--bg2));
  color: var(--text);
}

/* Remove top padding a bit */
.block-container{
  padding-top: 2.0rem !important;
  max-width: 860px;
}

/* Title styling */
.royal-title{
  font-size: 2.35rem;
  font-weight: 800;
  letter-spacing: .5px;
  margin-bottom: .2rem;
  background: linear-gradient(90deg, var(--gold2), var(--gold));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.royal-subtitle{
  color: var(--muted);
  font-size: 1.02rem;
  margin-bottom: 1.2rem;
}

/* Card wrapper */
.royal-card{
  background: linear-gradient(180deg, rgba(15,20,38,.92), rgba(16,26,51,.78));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 10px 18px;
  box-shadow: var(--shadow);
}

/* Thin golden divider */
.royal-divider{
  height: 1px;
  width: 100%;
  margin: 12px 0 14px 0;
  background: linear-gradient(90deg, transparent, rgba(212,175,55,.65), transparent);
}

/* Chat bubbles */
[data-testid="stChatMessage"]{
  background: transparent !important;
  border: none !important;
  padding: 0.35rem 0 !important;
}

[data-testid="stChatMessage"] > div{
  background: linear-gradient(180deg, rgba(15,20,38,.85), rgba(16,26,51,.70));
  border: 1px solid rgba(212,175,55,.20);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 35px rgba(0,0,0,.35);
}

/* Different accents for user vs assistant */
[data-testid="stChatMessage"][aria-label="chat message user"] > div{
  border: 1px solid rgba(212,175,55,.35);
}

[data-testid="stChatMessage"][aria-label="chat message assistant"] > div{
  border: 1px solid rgba(130,90,255,.25);
}

/* Chat input styling */
[data-testid="stChatInput"]{
  background: transparent;
}

[data-testid="stChatInput"] textarea{
  background: rgba(15,20,38,.85) !important;
  border: 1px solid rgba(212,175,55,.30) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
  box-shadow: 0 10px 35px rgba(0,0,0,.35) !important;
}

[data-testid="stChatInput"] textarea:focus{
  border: 1px solid rgba(212,175,55,.55) !important;
}

/* Buttons (sidebar / clear chat) */
.stButton>button{
  width: 100%;
  background: linear-gradient(90deg, rgba(212,175,55,.22), rgba(130,90,255,.18));
  border: 1px solid rgba(212,175,55,.35);
  color: var(--text);
  border-radius: 14px;
  padding: .55rem .8rem;
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}

.stButton>button:hover{
  border: 1px solid rgba(212,175,55,.60);
  transform: translateY(-1px);
}

/* Sidebar styling */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(8,10,16,.92), rgba(10,14,24,.85));
  border-right: 1px solid rgba(212,175,55,.20);
}

section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

.small-muted{
  color: var(--muted);
  font-size: 0.92rem;
}

/* Code blocks */
pre{
  border: 1px solid rgba(212,175,55,.18) !important;
  border-radius: 14px !important;
  background: rgba(10,14,24,.75) !important;
}

/* Links */
a{
  color: var(--gold2) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# HEADER UI
# ----------------------------
st.markdown('<div class="royal-title">👑  C++ Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="royal-subtitle">Ask any question about your <b>C++ Introduction</b> notes. '
    'This bot retrieves the most relevant sections from your file.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

load_dotenv()

FILE_PATH = "C++_Introduction.txt"  # keep in same folder as this .py

# ----------------------------
# VECTOR STORE BUILD
# ----------------------------
@st.cache_resource
def build_vectorstore():
    # 1) Load text
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    documents = loader.load()

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(documents)

    # 3) Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-miniLM-L6-v2"
    )

    # 4) FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Safety check
if not os.path.exists(FILE_PATH):
    st.error(f"File not found: **{FILE_PATH}**. Put it in the same folder as this app.")
    st.stop()

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------------------------
# SIDEBAR (ROYAL INFO PANEL)
# ----------------------------
with st.sidebar:
    st.markdown("## 👑 Control Panel")
    st.markdown('<div class="small-muted">Royal theme • Fast local retrieval • No LLM used</div>', unsafe_allow_html=True)
    st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📄 Knowledge Source")
    st.write(FILE_PATH)

    st.markdown("### 🧠 Retrieval Settings")
    k = st.slider("Top K chunks", min_value=1, max_value=8, value=3, step=1)
    retriever.search_kwargs = {"k": k}

    show_sources = st.toggle("Show sources (chunks)", value=True)

    st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# CHAT STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# CHAT INPUT
# ----------------------------
user_query = st.chat_input("Type your C++ question… (e.g., What is OOP in C++?)")

if user_query:
    # Save & show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve docs (new LangChain method)
    docs = retriever.invoke(user_query)

    # Build response
    if not docs:
        answer = "I couldn't find anything relevant in your notes. Try rephrasing your question."
    else:
        # Simple "answer" as extracted content (retrieval-based)
        extracted = "\n\n".join([d.page_content.strip() for d in docs])
        answer = f"Here’s what I found in your notes:\n\n{extracted}"

        if show_sources:
            meta = "\n".join(
                [f"- Source {i+1}: page={d.metadata.get('page', 'N/A')}, file={d.metadata.get('source', FILE_PATH)}"
                 for i, d in enumerate(docs)]
            )
            answer += "\n\n---\n**Sources:**\n" + meta

    # Save & show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Footer note
st.markdown('<div class="royal-divider"></div>', unsafe_allow_html=True)

st.caption("Tip: This version is retrieval-only. If you want *real AI answers*, tell me if you want Gemini / OpenAI / Ollama integration.")
