"""Part A · Step 2 test — retrieval only, no LLM, no API key.

Shows exactly what the vector store hands back for a question: the 'R' in RAG.
The 'G' (Claude reads these chunks and answers) arrives in Step 3 as rag.py.

    python query.py "what is tracking error"
    python query.py "what is tracking error" "how to calculate sharpe ratio"
"""
import sys
from pathlib import Path
import chromadb

HERE = Path(__file__).parent
client = chromadb.PersistentClient(path=str(HERE / "chroma_db"))
col = client.get_or_create_collection("investment_kb")


def search(questions, k=3):
    # Accept one question (a str) OR several (a list). Normalise to a list so the
    # rest of the function treats both cases identically.
    if isinstance(questions, str):
        questions = [questions]
    # ONE batched call embeds all questions and searches for each at once.
    # Results are grouped per question: res["documents"][i] is question i's hits.
    res = col.query(query_texts=questions, n_results=k)
    for qi, question in enumerate(questions):
        print(f"\nQ: {question}")
        # distance = how far a chunk sits from the question (smaller = more relevant).
        hits = zip(res["documents"][qi], res["metadatas"][qi], res["distances"][qi])
        for rank, (doc, meta, dist) in enumerate(hits, 1):
            print(f"  #{rank}  distance={dist:.3f}  [source: {meta['source']}, v{meta.get('version', '?')}]")
            print(f"      {doc}")


if __name__ == "__main__":
    # Each command-line argument is ONE question, so multi-word questions need quotes.
    # No argument -> fall back to a single default question.
    questions = sys.argv[1:] or ["What is our official definition of tracking error?"]
    search(questions)
