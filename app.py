import os
import gradio as gr
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# ==============================
# Environment setup
# ==============================
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"

PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
os.makedirs(PERSIST_DIR, exist_ok=True)

retriever = None
qa_chain = None

# ==============================
# Indexing
# ==============================
def index_wikipedia(topics, max_pages, chunk_size, chunk_overlap):
    try:
        topics = [t.strip() for t in topics.split("\n") if t.strip()]
        if not topics:
            return "⚠️ Please enter at least one topic."
        
        docs = fetch_wikipedia_pages(topics, max_pages_per_topic=max_pages)
        if not docs:
            return "⚠️ No pages fetched. Try different topics."
        
        build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        global retriever, qa_chain
        retriever = get_retriever(k=3)
        qa_chain = build_qa_chain(retriever)
        
        return f"✅ Indexed {len(docs)} pages successfully! You can now ask questions."
    except Exception as e:
        return f"❌ Indexing failed: {e}"

import os
import gradio as gr
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_pages
from retriever import build_or_load_vectorstore, get_retriever
from qa_chain import build_qa_chain
from utils import format_sources

# ==============================
# Environment setup
# ==============================
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMADB_TELEMETRY"] = "OFF"

PERSIST_DIR = os.getenv("PERSIST_DIR", "vectorstore")
os.makedirs(PERSIST_DIR, exist_ok=True)

retriever = None
qa_chain = None

# ==============================
# Indexing
# ==============================
def index_wikipedia(topics, max_pages, chunk_size, chunk_overlap):
    try:
        topics = [t.strip() for t in topics.split("\n") if t.strip()]
        if not topics:
            return "⚠️ Please enter at least one topic."
        
        docs = fetch_wikipedia_pages(topics, max_pages_per_topic=max_pages)
        if not docs:
            return "⚠️ No pages fetched. Try different topics."
        
        build_or_load_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        global retriever, qa_chain
        retriever = get_retriever(k=3)
        qa_chain = build_qa_chain(retriever)
        
        return f"✅ Indexed {len(docs)} pages successfully! You can now ask questions."
    except Exception as e:
        return f"❌ Indexing failed: {e}"

# ==============================
# Chat Interface
# ==============================
def chat_with_model(message, history):
    if not message.strip():
        return history, ""
    
    if qa_chain is None:
        history.append((message, "⚠️ Please index Wikipedia pages first using the left panel."))
        return history, ""
    
    try:
        result = qa_chain({"query": message})
        answer = result.get("result", "Sorry, I couldn't generate an answer.")
        docs = result.get("source_documents", [])
        
        # Check if answer indicates no information
        if "don't have information" in answer.lower() or "not in" in answer.lower():
            response = f"{answer}\n\n💡 **Tip:** Try indexing more topics or rephrasing your question."
        else:
            sources = format_sources(docs)
            response = f"{answer}\n\n📚 **Sources:**\n{sources}"
        
        history.append((message, response))
        return history, ""
    except Exception as e:
        history.append((message, f"❌ Error: {e}"))
        return history, ""

def clear_chat():
    return []

# ==============================
# Gradio UI
# ==============================
with gr.Blocks(theme=gr.themes.Soft(), title="Offline Wikipedia RAG") as demo:
    gr.Markdown(
        """
        # 🧠 Chat with Offline Wikipedia
        ### Powered by Local Phi-2 Model (100% Offline)
        
        **How to use:**
        1. Enter topics in the left panel (one per line)
        2. Click "📚 Fetch & Index Wikipedia"
        3. Wait for indexing to complete
        4. Ask questions in the chat!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Wikipedia Indexing Settings")
            topics = gr.Textbox(
                label="Enter topics (one per line)",
                value="Artificial intelligence\nPython programming\nBlack holes\nQuantum computing",
                lines=6,
                placeholder="Enter topics to index..."
            )
            max_pages = gr.Slider(
                1, 10, value=3, step=1, 
                label="Max pages per topic",
                info="More pages = better answers but slower indexing"
            )
            chunk_size = gr.Slider(
                100, 2000, value=500, step=100, 
                label="Chunk size",
                info="Size of text chunks (300-800 recommended)"
            )
            chunk_overlap = gr.Slider(
                0, 500, value=50, step=10, 
                label="Chunk overlap",
                info="Overlap between chunks (prevents info loss)"
            )
            index_btn = gr.Button("📚 Fetch & Index Wikipedia", variant="primary")
            index_output = gr.Textbox(label="Indexing Status", lines=3)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat Interface")
            chatbot = gr.Chatbot(
                height=500,
                show_label=False
            )
            msg = gr.Textbox(
                label="Type your question here...",
                placeholder="Ask anything about the indexed topics...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    # Event handlers
    index_btn.click(
        index_wikipedia,
        inputs=[topics, max_pages, chunk_size, chunk_overlap],
        outputs=index_output,
    )
    
    msg.submit(chat_with_model, [msg, chatbot], [chatbot, msg])
    submit_btn.click(chat_with_model, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    # Get port from environment (for cloud deployment) or use 7860 for local
    port = int(os.environ.get("PORT", 7860))
    
    demo.launch(
        share=False,  # Set to True for temporary public URL
        server_name="0.0.0.0",  # Allow external access
        server_port=port,
        show_api=True  # Enable API documentation
    )