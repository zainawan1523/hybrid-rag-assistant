import os
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_prompt

# .env file se OPENAI_API_KEY parhna (key code mein kabhi na likhein)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY nahi mili. Isi folder mein .env file banayein.")

client = OpenAI(api_key=api_key)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # .env mein model badal sakte hain


# ---------- Step 4: Generation (prompt => LLM => jawab) ----------
def generate_answer(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # R: retriever tayyar karo (yeh local chalta hai, API key nahi lagti)
    chunks = load_and_chunk("notes.pdf", chunk_size=500, chunk_overlap=50)
    retriever = HybridRetriever(chunks)

    while True:
        question = input("\nApna sawal likhein (band karne ke liye 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        results = retriever.search(question, top_k=3)   # R: Retrieval
        prompt = build_prompt(question, results)        # A: Augmentation
        answer = generate_answer(prompt)                # G: Generation

        print("\n" + "=" * 70)
        print("JAWAB:")
        print(answer)
        print("\nSources:", ", ".join(
            f"Page {doc.metadata.get('page', '?')}" for doc, _ in results
        ))