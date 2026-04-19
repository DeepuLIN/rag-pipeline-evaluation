import gradio as gr

from src.rag.answer import answer_question


def convert_history(history):
    history = history or []
    rag_history = []

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item

            if user_msg:
                rag_history.append({"role": "user", "content": str(user_msg)})

            if assistant_msg:
                rag_history.append({"role": "assistant", "content": str(assistant_msg)})

        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content is not None:
                rag_history.append({"role": role, "content": str(content)})

    return rag_history


def chat_fn(message, history):
    rag_history = convert_history(history)
    answer, _ = answer_question(message, rag_history)

    if isinstance(answer, list):
        try:
            answer = answer[0].get("text", str(answer))
        except Exception:
            answer = str(answer)

   
    if isinstance(answer, str) and answer.startswith("[{"):
        import json
        try:
            parsed = json.loads(answer)
            answer = parsed[0].get("text", answer)
        except Exception:
            pass

    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Insurellm RAG Assistant",
    description="Ask questions about Insurellm using the internal knowledge base.",
)


if __name__ == "__main__":
    demo.launch()