"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

# TODO: Fill this in! Include a few labeled examples (k-shot) in the system prompt.
YOUR_SYSTEM_PROMPT = """You are a customer support ticket classifier for an online store.

Classify each customer message into exactly one of these five categories:
billing, shipping, returns, technical, other.

Category definitions:
- billing: Charges, payments, invoices, and billing errors.
- shipping: Deliveries, tracking, missing packages, and shipping delays.
- returns: Product returns, exchanges, and refund requests.
- technical: Website, application, account, and software errors.
- other: Messages that do not fit any of the categories above.

Learn from these examples:

Customer: "I received an incorrect invoice."
Category: billing

Customer: "The tracking number has not updated."
Category: shipping

Customer: "Can I exchange this shirt for another size?"
Category: returns

Customer: "I cannot log in to my account."
Category: technical

Customer: "Do you sell gift cards?"
Category: other

Now classify the user's message.

Output rules:
- Return only one category label.
- Use lowercase letters.
- Do not include explanations, punctuation, or extra text.
"""
CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
