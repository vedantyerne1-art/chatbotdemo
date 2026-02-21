import os
import time
import uuid
import sqlite3
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()

APP_TITLE = "C++ AI Assistant"
FILE_PATH = "C++_Introduction.txt"

DB_PATH = "chat_history.db"
TABLE_NAME = "messages"


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# PREMIUM DARK THEME
# ----------------------------
st.markdown(
    """
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
  font-size: 2.25rem;
  font-weight: 850;
  letter-spacing: .3px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  margin-bottom: .25rem;
}
.subtitle{
  color: var(--muted);
  margin-bottom: 1.05rem;
}
.hero{
  background: var(--card);
  border:1px solid var(--border);
  border-radius:18px;
  padding:16px;
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
  font-weight:650;
}
.stButton>button:hover{ opacity:0.92; }

section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0a0f20, #121733);
}
section[data-testid="stSidebar"] *{ color:var(--text) !important; }

.small{
  color: var(--muted);
  font-size: 0.92rem;
}
hr{
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(120,90,255,.55), transparent);
  margin: 12px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f'<div class="title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions from your C++ Introduction notes. Chat is saved automatically.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Local Retrieval • Fast • Lightweight • History Saved in SQLite</div>', unsafe_allow_html=True)


# ----------------------------
# SQLITE HELPERS
# ----------------------------
def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def db_init():
    with db_connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

def db_save_message(session_id: str, role: str, content: str):
    with db_connect() as conn:
        conn.execute(
            f"INSERT INTO {TABLE_NAME} (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()

def db_get_sessions(limit: int = 30):
    with db_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT session_id, MAX(created_at) AS last_time, COUNT(*) AS msg_count
            FROM {TABLE_NAME}
            GROUP BY session_id
            ORDER BY last_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows  # [(session_id, last_time, msg_count), ...]

def db_load_session_messages(session_id: str):
    with db_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT role, content, created_at
            FROM {TABLE_NAME}
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return rows  # [(role, content, created_at), ...]

def db_delete_session(session_id: str):
    with db_connect() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME} WHERE session_id = ?", (session_id,))
        conn.commit()


# ----------------------------
# VECTOR STORE BUILD
# ----------------------------
@st.cache_resource
def build_vector():
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ----------------------------
# SAFETY CHECKS
# ----------------------------
db_init()

if not os.path.exists(FILE_PATH):
    st.error(f"❌ File not found: **{FILE_PATH}**. Put it in the same folder as this app.")
    st.stop()

vectorstore = build_vector()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "...", "content": "..."}

def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

def load_chat(session_id: str):
    rows = db_load_session_messages(session_id)
    st.session_state.session_id = session_id
    st.session_state.messages = [{"role": r[0], "content": r[1]} for r in rows]
    st.rerun()


# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("Settings")

    k = st.slider("Top Results (k)", 1, 8, 3)
    retriever.search_kwargs = {"k": k}

    show_sources = st.toggle("Show Sources", value=True)
    typing_effect = st.toggle("Typing effect", value=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("Chat History")
    st.markdown('<div class="small">Load previous chats from SQLite.</div>', unsafe_allow_html=True)

    sessions = db_get_sessions(limit=40)
    session_labels = []
    session_map = {}  # label -> session_id
    for sid, last_time, msg_count in sessions:
        short = sid.split("-")[0]
        label = f"{short} • {last_time} • {msg_count} msgs"
        session_labels.append(label)
        session_map[label] = sid

    selected = st.selectbox("Saved sessions", options=["(current session)"] + session_labels)

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ New Chat"):
            start_new_chat()

    with colB:
        if selected != "(current session)" and st.button("📂 Load"):
            load_chat(session_map[selected])

    if selected != "(current session)":
        if st.button("🗑️ Delete Selected Session"):
            db_delete_session(session_map[selected])
            # If deleting currently loaded session, start new
            if st.session_state.session_id == session_map[selected]:
                start_new_chat()
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption(f"Active session: {st.session_state.session_id.split('-')[0]}")


# ----------------------------
# RENDER PREVIOUS MESSAGES
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ----------------------------
# CHAT INPUT + RESPONSE
# ----------------------------
query = st.chat_input("Ask your C++ question...")

if query:
    # Save user message in UI + DB
    st.session_state.messages.append({"role": "user", "content": query})
    db_save_message(st.session_state.session_id, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve docs
    docs = retriever.invoke(query)

    if not docs:
        answer_text = "I couldn't find relevant content in your notes. Try rephrasing your question."
        sources_text = ""
    else:
        combined = "\n\n".join([d.page_content.strip() for d in docs])
        answer_text = f"Here’s what I found in your notes:\n\n{combined}"

        sources_text = ""
        if show_sources:
            sources_text = "\n\n---\n**Sources:**\n" + "\n".join(
                [f"- Chunk {i+1} (file: {os.path.basename(d.metadata.get('source', FILE_PATH))})" for i, d in enumerate(docs)]
            )

    final_answer = answer_text + sources_text

    # Assistant message render (typing optional)
    with st.chat_message("assistant"):
        if typing_effect:
            placeholder = st.empty()
            buf = ""
            for ch in final_answer:
                buf += ch
                placeholder.markdown(buf)
                time.sleep(0.0015)
        else:
            st.markdown(final_answer)

    # Save assistant message in UI + DB
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    db_save_message(st.session_state.session_id, "assistant", final_answer)
