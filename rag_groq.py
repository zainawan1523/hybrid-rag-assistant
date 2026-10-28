import os
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_prompt

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY nahi mili. Isi folder mein .env file banayein.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def generate_answer(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
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