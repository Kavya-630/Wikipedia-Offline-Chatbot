
# 🧠 Offline Wikipedia RAG (Retrieval-Augmented Generation)

A fully **offline** question-answering (RAG) system built using **LangChain**, **Gradio**, and a **local Phi-2 model (Llama.cpp)**.  
This app allows you to fetch Wikipedia content, build a local vector database, and query the information — **without requiring internet access** for the model or retriever.

---

## 🚀 Features

- 🧩 **Fetch and index Wikipedia pages** by topic (configurable number of pages per topic)
- 🔍 **Split and embed text** into a local Chroma vectorstore using Sentence Transformers
- 🤖 **Local reasoning** via **Phi-2 LLM (Llama.cpp)** — 100% offline inference
- 💬 **Interactive Gradio UI** for chat-based QA
- 🧠 **Anti-hallucination prompt** ensuring model answers *only* from retrieved context
- 🧰 Modular architecture:
  - `wiki_loader.py` → fetch and clean Wikipedia pages  
  - `retriever.py` → chunking, embeddings, vectorstore  
  - `qa_chain.py` → question answering with local or HuggingFace model  
  - `app.py` → Gradio front-end  
  - `utils.py` → helper functions for clean outputs  

---

## 🧩 Architecture Overview

Wikipedia Pages ──► LangChain Document Loader
│
▼
RecursiveCharacterTextSplitter ──► SentenceTransformer Embeddings
│
▼
Local Chroma Vectorstore (persisted to disk)
│
▼
Retriever ──► LLM (Phi-2 via LlamaCpp)
│
▼
Question Answer + Source Documents

##Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

##Install Dependencies
pip install -r requirements.txt

##Model Setup (Local Phi-2)
Download a Phi-2 GGUF model compatible with llama-cpp-python.
Link to download the model from hugging face https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q4_K_M.gguf.
After downloading the model connect to your code in qa_chain.py file 

##Run the App
python app.py
Then open the Gradio interface in your browser (default: http://localhost:7860)
App looks like this
<img width="760" height="512" alt="image" src="https://github.com/user-attachments/assets/bd7284e2-6cd8-4ad3-8b5e-b8297575cf69" />

##💻 How to Use
1. Enter topics to index in the left panel (one per line)
  e.g.
  Artificial Intelligence
  Quantum Computing
  Black Holes
  Python Programming
2. Click “📚 Fetch & Index Wikipedia”
3. Wait until indexing completes (you’ll see ✅ success message)
4. Start asking questions in the chat panel!

<img width="1129" height="600" alt="image" src="https://github.com/user-attachments/assets/b2cff878-d5e6-48bf-90c5-adc67893d57b" />
