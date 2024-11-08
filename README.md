# Langchain RAG Chatbot

This project uses the Langchain framework along with the Retrieval-Augmented Generation (RAG) concept to build a chatbot on Streamlit. 

### Overview
In this repository, we use a locally fine-tuned model (`distilgpt2`) on an extensive QnA dataset containing over 20,000 questions and responses. This code structure supports using your own models without relying on cloud-based GPT services or Ollama's local server.

### Steps to Get Started

1. **Clone the repository.**

2. **Generate Embeddings**  
   Run `Embeddings_store.py` to create embeddings for the document or PDF you want the chatbot to reference through RAG.

3. **Helper Script for Embeddings**  
   The script `sbert.py` provides helper functions for embeddings, but running this file separately is not required.

4. **Run the Chatbot**  
   Launch the chatbot by opening a terminal and running:
   ```bash
   streamlit run app.py
