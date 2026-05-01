import gradio as gr
from my_service import handle_user_query

# Creating a chat history to display on UI.
chat_history = []

# Creating Guardrails.
def apply_guardrails(user_input: str) -> str:
    restricted_topics = ["cats", "dogs", "horoscope", "zodiac", "taylor swift"]

    for topic in restricted_topics:
        if topic.lower() in user_input.lower():
            return "Sorry, I cannot respond to that topic."

    return None


# Creating a chat function for Gradio UI served as communication
# between the user and the Chatbox.
def chat(user_input, history):
    if history is None:
        history = []

    # Creating Guardrails.
    blocked = apply_guardrails(user_input)
    if blocked:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": blocked})
        return history, history

    # Creating a call service.
    response = handle_user_query(user_input)
    response = f"🤖 AI Assistant: {response}"

    # Creating a history log to display on the UI.
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    return history, history


# Creating a Gradio UI simple to use.
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Kevin's AI Chatbot")
    gr.Markdown("Ask me anything except restricted topics. :)")

    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message here...",
            label="Your Message"
        )
        send_btn = gr.Button("Send")

    # Creating a Send button on the UI.
    send_btn.click(
        chat,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

    # Creating an Enter button to send input from user.
    user_input.submit(
        chat,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

# Running the app. The prompt will show a link.
# Copy the link and paste it on a browser to access the app.
if __name__ == "__main__":
    demo.launch()


