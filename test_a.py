"""Part A · Step 3 smoke test — non-interactive, runs the key checks in one go.

Test 3 is the important one: an out-of-scope question the model must REFUSE,
proving refusal lives in generation, not retrieval.
"""
from rag import answer
from llm import describe

TESTS = [
    "What was the funded status, and as of when?",           # PDF narrative (text retrieval)
    "What is our official definition of tracking error?",    # governed metric + version
    "What is the portfolio manager's dog's name?",           # must refuse
]

if __name__ == "__main__":
    print(f"=== Part A smoke test [{describe()}] ===")
    for q in TESTS:
        print(f"\nQ: {q}\nA: {answer(q)}")
