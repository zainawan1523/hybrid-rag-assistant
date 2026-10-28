import os
import tempfile
import streamlit as st

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_context
from smart_rag import PERSONA, AUDIENCES, RULES, ask_llm

st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="📚")


# ---------- Retriever ko cache karna (har sawal par dobara na bane) ----------
@st.cache_resource(show_spinner="PDF parh raha hoon aur chunks bana raha hoon...")
def get_retriever(pdf_bytes, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    chunks = load_and_chunk(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    os.remove(path)
    return HybridRetriever(chunks), len(chunks)


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader("Apni PDF upload karein", type="pdf")
    st.caption("Upload na karein to notes.pdf use hogi.")

    audience = st.selectbox(
        "Audience (aap kaun hain?)",
        list(AUDIENCES),
        format_func=lambda a: {"ai_engineer": "AI Engineer",
                               "student": "Student",
                               "manager": "Manager"}.get(a, a),
    )
    top_k = st.slider("Kitne chunks dhoondein (top_k)", 1, 5, 3)
    chunk_size = st.slider("Chunk size", 200, 1000, 500, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, step=25)

    if st.button("🗑️ Chat saaf karein"):
        st.session_state.messages = []


# ---------- PDF load ----------
if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    pdf_name = uploaded.name
elif os.path.exists("notes.pdf"):
    with open("notes.pdf", "rb") as f:
        pdf_bytes = f.read()
    pdf_name = "notes.pdf"
else:
    st.warning("Koi PDF nahi mili. Sidebar se PDF upload karein.")
    st.stop()

retriever, n_chunks = get_retriever(pdf_bytes, chunk_size, chunk_overlap)


# ---------- Header ----------
st.title("📚 Hybrid RAG Assistant")
st.caption(f"Document: **{pdf_name}**  |  Chunks: **{n_chunks}**  |  "
           f"Search: BM25 + Embeddings (RRF)")


# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources dekhein"):
                st.markdown(msg["sources"])


# ---------- Naya sawal ----------
question = st.chat_input("Apna sawal likhein...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            # R: Retrieval
            results = retriever.search(question, top_k=top_k)
            context = build_context(results)

            # A: Persona + Audience (sidebar se) + Rules
            system_msg = f"{PERSONA}\n\n{AUDIENCES[audience]}\n\n{RULES}"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]

            # G: Generation
            try:
                answer = ask_llm(messages)
            except Exception as e:
                answer = f"⚠️ LLM se jawab nahi aaya: `{e}`"

        st.markdown(answer)

        sources = "\n\n".join(
            f"**Page {doc.metadata.get('page', '?')}** (score {score:.4f})\n\n"
            f"> {doc.page_content[:300].strip()}..."
            for doc, score in results
        )
        with st.expander("📄 Sources dekhein"):
            st.markdown(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )