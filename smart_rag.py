import os
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_context

load_dotenv()
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ================= PATTERN 1: PERSONA (AI kaun hai) =================
PERSONA = """You are a senior AI/ML engineer and RAG specialist with 10 years of
experience. You explain things accurately and never make up facts."""


# ================= PATTERN 2: AUDIENCE PERSONA (user kaun hai) =================
# User ke baare mein pehle se pata hai, is liye har dafa poochne ki zaroorat nahi.
AUDIENCES = {
    "ai_engineer": """The user is an AI engineer. They already know Python, LLMs,
embeddings and vector databases. Be technical and concise. Use correct terms,
mention parameters and trade-offs, and give a short code snippet when it helps.""",

    "student": """The user is a beginner student learning AI. Use simple words,
explain any technical term, and give one everyday example.""",

    "manager": """The user is a non-technical manager. Avoid jargon and code.
Focus on what it is, why it matters for the business, and give a 2-3 line summary.""",
}

# Yahan user ki type set karein: "ai_engineer", "student" ya "manager"
AUDIENCE = os.getenv("AUDIENCE", "ai_engineer")


# ================= GROUNDING RULES (fallback ke saath) =================
RULES = """Rules:
1. First use the context. If the answer is there, end with "(Source: documents, Page X)".
2. If the context does not have the answer, do not refuse. Answer from your own
   knowledge and start with "[Not in documents - general knowledge]".
3. If it is something you cannot know (a private person, very recent events),
   say so honestly."""


def build_messages(question, context):
    # System message = Persona + Audience + Rules (yeh model ka "kirdar" set karta hai)
    system_msg = f"{PERSONA}\n\n{AUDIENCES[AUDIENCE]}\n\n{RULES}"

    # User message = retrieved chunks + sawal
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def ask_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY nahi mili. Isi folder mein .env file check karein.")
    if AUDIENCE not in AUDIENCES:
        raise SystemExit(f"AUDIENCE '{AUDIENCE}' ghalat hai. In mein se chunein: {list(AUDIENCES)}")

    chunks = load_and_chunk("notes.pdf", chunk_size=500, chunk_overlap=50)
    retriever = HybridRetriever(chunks)
    print(f"\nAudience: {AUDIENCE}  |  Model: {MODEL}")

    while True:
        question = input("\nApna sawal likhein (band karne ke liye 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        results = retriever.search(question, top_k=3)    # R: Retrieval
        context = build_context(results)
        messages = build_messages(question, context)     # A: Persona + Audience + Context
        answer = ask_llm(messages)                       # G: Generation

        print("\n" + "=" * 70)
        print("JAWAB:")
        print(answer)