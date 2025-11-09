import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# Check if running on Hugging Face Spaces
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None

if IS_HUGGINGFACE:
    # Use Hugging Face's free hosted model
    from langchain_community.llms import HuggingFaceHub
    
    def build_qa_chain(retriever):
        """QA chain using Hugging Face hosted models."""
        print("[INFO] Using Hugging Face hosted model (free)")
        
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={
                "temperature": 0.1,
                "max_length": 300
            }
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Context: {context}

Question: {question}

STRICT RULES:
1. If context doesn't contain the answer, say: "I don't have information about this in my indexed Wikipedia pages."
2. Only use facts from the context
3. Be concise and direct

Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("[INFO] ⚡ QA chain built successfully")
        return qa_chain

else:
    # Use local Phi-2 model
    from langchain_community.llms import LlamaCpp
    
    LLAMA_PATH = os.getenv(
        "LLAMA_MODEL_PATH",
        r"C:\Users\Navya sree\Downloads\Gen AI\offline-wiki-rag-Gradio\models\phi-2.Q4_K_M.gguf"
    )
    
    def build_qa_chain(retriever):
        """QA chain with local Phi-2 model."""
        print(f"[INFO] Using local model: {LLAMA_PATH}")
        
        llm = LlamaCpp(
            model_path=LLAMA_PATH,
            temperature=0.1,
            max_tokens=300,
            n_ctx=2048,
            verbose=False,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=0,
            repeat_penalty=1.2,
        )
        
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that ONLY answers based on the provided context from Wikipedia.

Context from Wikipedia:
{context}

Question: {question}

STRICT RULES:
1. If the context does NOT contain information to answer the question, respond EXACTLY with: "I don't have information about this in my indexed Wikipedia pages. Please try indexing more topics."
2. NEVER make up or invent information
3. ONLY use facts directly from the context above
4. If you can partially answer, say what you know and clearly state what's missing
5. Always be concise and direct

Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("[INFO] ⚡ Anti-hallucination QA chain built successfully")
        return qa_chain


# Test function
if __name__ == "__main__":
    print("✅ qa_chain.py loaded successfully")