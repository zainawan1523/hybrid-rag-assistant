from hybrid_chunked import load_and_chunk, HybridRetriever


# ---------- Step 3: Augmentation (chunks + sawal => prompt) ----------
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the given documents."
At the end, mention the page numbers you used, like (Page 4).

=== CONTEXT ===
{context}
=== END CONTEXT ===

Question: {question}
Answer:"""


def build_context(results):
    """Retrieved chunks ko ek text block mein jorna, har chunk ke saath uska source."""
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        parts.append(f"[Source {i} | Page {page}]\n{text}")
    return "\n\n".join(parts)


def build_prompt(question, results):
    """Context aur sawal ko template mein daal kar final prompt banana."""
    context = build_context(results)
    return PROMPT_TEMPLATE.format(context=context, question=question)


if __name__ == "__main__":
    # Retrieval (pehle wala kaam)
    chunks = load_and_chunk("notes.pdf", chunk_size=500, chunk_overlap=50)
    retriever = HybridRetriever(chunks)

    while True:
        question = input("\nApna sawal likhein (band karne ke liye 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        # R: relevant chunks dhoondo
        results = retriever.search(question, top_k=3)

        # A: chunks ko prompt se jorna
        prompt = build_prompt(question, results)

        print("\n" + "=" * 70)
        print("FINAL PROMPT (yeh LLM ko jayega):")
        print("=" * 70)
        print(prompt)