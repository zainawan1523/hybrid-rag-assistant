import os
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_chunked import load_and_chunk, HybridRetriever
from rag_augment import build_context

load_dotenv()
client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ================= PROMPT PATTERNS =================
# Har pattern mein {context} aur {question} ki jagah asal data aata hai.

PATTERNS = {
    # 1. Basic: sirf context aur sawal, koi khaas hidayat nahi
    "1_basic": """Context:
{context}

Question: {question}
Answer:""",

    # 2. Strict / Grounded: sirf context se jawab, warna "I don't know"
    "2_strict": """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the given documents."

Context:
{context}

Question: {question}
Answer:""",

    # 3. Role-based: model ko ek kirdar (teacher) dena
    "3_role": """You are a friendly teacher explaining AI concepts to a beginner student.
Use simple words and one everyday example. Use only the context below.

Context:
{context}

Question: {question}
Answer:""",

    # 4. Few-shot: pehle ek misaal dikhana ke jawab kaisa hona chahiye
    "4_few_shot": """Answer in the same style as the example. Use only the context.

Example:
Question: What is an embedding?
Answer: An embedding is a list of numbers that represents the meaning of text. (Page 3)

Context:
{context}

Question: {question}
Answer:""",

    # 5. Chain-of-Thought: pehle soch ke steps, phir final jawab
    "5_step_by_step": """Use only the context below. First, list which sources are relevant and why.
Then think step by step. Finally write "Final Answer:" followed by a short answer.

Context:
{context}

Question: {question}""",

    # 6. Structured output: fixed format mein jawab
    "6_structured": """Use only the context below. Reply in exactly this format:

Definition: <one sentence>
Key points:
- <point 1>
- <point 2>
Source pages: <page numbers>

Context:
{context}

Question: {question}""",

    # 7. Inline citations: har jumle ke saath source number
    "7_citations": """Use only the context below. After every sentence, add the source it came
from in square brackets, like [Source 1]. Do not write any sentence without a source.

Context:
{context}

Question: {question}
Answer:""",

    # 8. Fallback: pehle documents, na milay to apni knowledge (label ke saath)
    "8_fallback": """Answer the question in two steps.

Step 1: Check the context below. If it contains the answer, answer from it and
end with "(Source: documents, Page X)".

Step 2: If the context does NOT contain the answer, do not refuse. Answer from your
own general knowledge and reasoning instead, and start your answer with
"[Not in documents - general knowledge]".
If the question is about something you cannot know (for example a private person,
or very recent events), say that honestly and explain what information would be needed.

Context:
{context}

Question: {question}
Answer:""",
}


def ask_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def run_pattern(name, question, context):
    prompt = PATTERNS[name].format(context=context, question=question)
    answer = ask_llm(prompt)
    print("\n" + "=" * 70)
    print(f"PATTERN: {name}   |   prompt length: {len(prompt)} chars")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY nahi mili. Isi folder mein .env file check karein.")

    chunks = load_and_chunk("notes.pdf", chunk_size=500, chunk_overlap=50)
    retriever = HybridRetriever(chunks)
    names = list(PATTERNS)

    while True:
        question = input("\nApna sawal likhein (band karne ke liye 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("\nPatterns:")
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
        choice = input("Number likhein, ya sab compare karne ke liye 'all': ").strip().lower()

        # Retrieval sirf ek dafa, taake har pattern ko bilkul same chunks milein
        results = retriever.search(question, top_k=3)
        context = build_context(results)

        if choice == "all":
            selected = names
        elif choice.isdigit() and 1 <= int(choice) <= len(names):
            selected = [names[int(choice) - 1]]
        else:
            print("Ghalat number.")
            continue

        for name in selected:
            run_pattern(name, question, context)