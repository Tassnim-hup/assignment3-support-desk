"""Tool calling — order lookup.

Task: the model must decide to call the `get_order_status` tool with the order id it
finds in the user's message, then use the returned status in a short reply.

Fill in YOUR_SYSTEM_PROMPT and the tool schema TODO.
Run:  python tool_calling.py
"""
from ollama import chat

MODEL = "llama3.1:8b"

FAKE_DB = {
    "4412": "shipped, arriving Friday",
    "9001": "processing",
}


def get_order_status(order_id: str) -> str:
    return FAKE_DB.get(order_id, "not found")


# TODO: describe the tool so the model knows when and how to call it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status of a customer order using its order ID. Call this function whenever a user asks about an order's status, shipping progress, or delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The exact order ID extracted from the customer's message, passed as a string without adding or changing digits."
                    },
                },
                "required": ["order_id"],
            },
        },
    }
]

# TODO: Fill this in! Tell the model to look up the order before answering.

YOUR_SYSTEM_PROMPT = """You are an order support assistant for an online store.

When a customer asks about the status, shipment, or delivery of an order, you MUST use the available get_order_status tool before providing an answer.

Instructions:
1. Identify the exact order ID in the customer's message.
2. Call get_order_status with that order ID as a string.
3. Do not guess or invent order statuses.
4. If a tool result is provided, use only that result to describe the order status.
5. If the order is not found, clearly explain that the order could not be located.

Do not answer an order-status question from your own knowledge. Always rely on the tool result.
"""



def run() -> bool:
    message = "Hi, can you check on order 4412 for me?"
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tools=TOOLS,
        options={"temperature": 0.0},
    )
    calls = resp.message.tool_calls or []
    if not calls:
        print("[FAIL] model did not call the tool")
        return False
    call = calls[0]
    order_id = str(call.function.arguments.get("order_id"))
    ok = order_id == "4412"
    print(f"[{'PASS' if ok else 'FAIL'}] tool called with order_id={order_id!r}")
    if ok:
        print("tool result:", get_order_status(order_id))
    return ok


if __name__ == "__main__":
    run()
