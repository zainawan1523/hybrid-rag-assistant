import os
import html
import tempfile
import streamlit as st

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_context
from smart_rag import PERSONA, AUDIENCES, RULES, ask_llm

st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="📚", layout="centered")


# ======================= CUSTOM CSS =======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, input, textarea, button {
    font-family: 'Inter', sans-serif !important;
}

/* Upar wali header bar transparent */
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; padding-bottom: 6rem; max-width: 820px; }

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #0f172a 100%);
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 26px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.hero h1 {
    margin: 0 0 6px 0; padding: 0;
    font-size: 1.9rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;
}
.hero p { margin: 0 0 16px 0; color: #c7d2fe; font-size: 0.95rem; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #e0e7ff; font-size: 0.78rem; font-weight: 500;
    padding: 5px 12px; border-radius: 999px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
.side-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #818cf8; margin: 18px 0 6px 0;
}

/* ---------- Chat bubbles ---------- */
div[data-testid="stChatMessage"] {
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
div[data-testid="stChatInput"] > div {
    border-radius: 14px !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
}

/* ---------- Sources ---------- */
div[data-testid="stExpander"] details {
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 10px !important;
    background: #0f172a;
}
.src {
    border-left: 3px solid #6366f1;
    background: rgba(99, 102, 241, 0.06);
    border-radius: 6px;
    padding: 10px 12px; margin-bottom: 10px;
}
.src-head { display: flex; justify-content: space-between; margin-bottom: 4px; }
.src-page { color: #a5b4fc; font-weight: 600; font-size: 0.82rem; }
.src-score { color: #64748b; font-size: 0.75rem; font-family: monospace; }
.src-text { color: #cbd5e1; font-size: 0.85rem; line-height: 1.5; }

/* ---------- Empty state ---------- */
.empty {
    text-align: center; color: #64748b; padding: 30px 0 10px 0; font-size: 0.95rem;
}
.empty b { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# ======================= RETRIEVER (cache) =======================
@st.cache_resource(show_spinner="Reading PDF and building chunks...")
def get_retriever(pdf_bytes, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    chunks = load_and_chunk(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    os.remove(path)
    return HybridRetriever(chunks), len(chunks)


AUDIENCE_LABELS = {"ai_engineer": "AI Engineer", "student": "Student", "manager": "Manager"}


# ======================= SIDEBAR =======================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown('<div class="side-title">Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your PDF", type="pdf",
                                label_visibility="collapsed")
    st.caption("No file uploaded? notes.pdf will be used.")

    st.markdown('<div class="side-title">Audience</div>', unsafe_allow_html=True)
    audience = st.selectbox("Who are you?", list(AUDIENCES),
                            format_func=lambda a: AUDIENCE_LABELS.get(a, a))

    st.markdown('<div class="side-title">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", 1, 5, 3)
    chunk_size = st.slider("Chunk size", 200, 1000, 500, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, step=25)

    st.markdown('<div class="side-title">Chat</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []


# ======================= PDF LOAD =======================
if uploaded is not None:
    pdf_bytes, pdf_name = uploaded.getvalue(), uploaded.name
elif os.path.exists("notes.pdf"):
    with open("notes.pdf", "rb") as f:
        pdf_bytes = f.read()
    pdf_name = "notes.pdf"
else:
    st.warning("No PDF found. Please upload one from the sidebar.")
    st.stop()

retriever, n_chunks = get_retriever(pdf_bytes, chunk_size, chunk_overlap)


# ======================= HERO HEADER =======================
st.markdown(f"""
<div class="hero">
  <h1>📚 Hybrid RAG Assistant</h1>
  <p>Ask questions about your documents and get answers backed by sources.</p>
  <div class="chips">
    <span class="chip">📄 {html.escape(pdf_name)}</span>
    <span class="chip">🧩 {n_chunks} chunks</span>
    <span class="chip">🔎 BM25 + Embeddings (RRF)</span>
    <span class="chip">👤 {AUDIENCE_LABELS.get(audience, audience)}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_sources(results):
    cards = []
    for doc, score in results:
        page = doc.metadata.get("page", "?")
        text = html.escape(doc.page_content[:300].strip().replace("\n", " "))
        cards.append(
            f'<div class="src"><div class="src-head">'
            f'<span class="src-page">Page {page}</span>'
            f'<span class="src-score">RRF {score:.4f}</span></div>'
            f'<div class="src-text">{text}…</div></div>'
        )
    return "".join(cards)


AVATARS = {"user": "🧑‍💻", "assistant": "🤖"}

# ======================= CHAT HISTORY =======================
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div class="empty">Type a question below to get started.<br>'
                'For example: <b>What is chunk overlap?</b></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources ({msg.get('n_sources', '')})"):
                st.markdown(msg["sources"], unsafe_allow_html=True)


# ======================= NEW QUESTION =======================
question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            # R: Retrieval
            results = retriever.search(question, top_k=top_k)
            context = build_context(results)

            # A: Persona + Audience + Rules
            system_msg = f"{PERSONA}\n\n{AUDIENCES[audience]}\n\n{RULES}"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]

            # G: Generation
            try:
                answer = ask_llm(messages)
            except Exception as e:
                answer = f"⚠️ Could not get a response from the LLM: `{e}`"

        st.markdown(answer)
        sources = render_sources(results)
        with st.expander(f"📄 Sources ({len(results)})"):
            st.markdown(sources, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "n_sources": len(results)}
    )