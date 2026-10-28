import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


# ---------- Step 1: PDF load + chunking ----------
def load_and_chunk(pdf_path, chunk_size=500, chunk_overlap=50):
    documents = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # Har chunk ko ek id de dein (debugging aur sources dikhane ke liye)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Pages: {len(documents)}  |  Chunks: {len(chunks)}")
    return chunks


# ---------- Step 2: Hybrid retriever on chunks ----------
class HybridRetriever:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks                        # LangChain Document objects
        texts = [c.page_content for c in chunks]

        # Sparse: BM25 on chunk text
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

        # Dense: chunk embeddings (normalized => dot product = cosine similarity)
        self.model = SentenceTransformer(model_name)
        self.chunk_emb = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def _bm25_rank(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        ids = np.argsort(scores)[::-1][:k]
        # Jin chunks mein query ka koi word hi nahi, unhein vote na dein
        return [i for i in ids if scores[i] > 0]

    def _dense_rank(self, query, k):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.chunk_emb @ q
        return np.argsort(scores)[::-1][:k]

    def search(self, query, top_k=3, candidates=20, rrf_k=60,
               w_dense=1.0, w_bm25=1.0):
        dense_ids = self._dense_rank(query, candidates)
        bm25_ids = self._bm25_rank(query, candidates)

        # Reciprocal Rank Fusion
        fused = {}
        for rank, idx in enumerate(dense_ids):
            fused[idx] = fused.get(idx, 0) + w_dense / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_ids):
            fused[idx] = fused.get(idx, 0) + w_bm25 / (rrf_k + rank + 1)

        best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], score) for i, score in best]


if __name__ == "__main__":
    chunks = load_and_chunk("notes.pdf", chunk_size=500, chunk_overlap=50)
    retriever = HybridRetriever(chunks)

    test_queries = [
        "What is chunk overlap?",                 # Chapter 4
        "What is FAISS?",                         # Chapter 3
        "What are the steps of a RAG pipeline?",  # Chapter 5
        "What is a Retriever in LangChain?",      # Chapter 6
        "What is the Transformer architecture?",  # Chapter 2
    ]

    for query in test_queries:
        print("=" * 70)
        print("SAWAL:", query)
        for doc, score in retriever.search(query, top_k=2):
            print(f"\n  Page: {doc.metadata['page']}  |  Chunk: {doc.metadata['chunk_id']}"
                  f"  |  RRF: {score:.4f}  (zyada score = behtar)")
            print("  ", doc.page_content[:150].replace("\n", " "), "...")
