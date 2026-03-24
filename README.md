# 🤖 Hybrid RAG Chatbot

A **Hybrid Retrieval-Augmented Generation (RAG) Chatbot** built using **LangChain, Chroma DB, and Llama3 (via Ollama)**.  
This system can answer questions from custom documents and also provide general AI responses when needed.

##  Features

-  Answer questions from PDF documents
-  Semantic search using vector embeddings
-  Hybrid AI (Document-based + General knowledge)
-  Local LLM using Ollama (Llama3)
-  Interactive UI with Streamlit
-  Automatic fallback when no relevant context is found

## How it works

User Query 
     ↓  
Retriever (Chroma DB)  
     ↓  
Relevant Document Chunks  
     ↓  
Llama3 (via Ollama)  
     ↓  
Final Answer  

If no relevant document is found → falls back to general LLM response.

## Key Highlights
Built a Hybrid RAG system combining retrieval + generation
Implemented fallback mechanism for better reliability
Uses local LLM (no API cost) 
Designed for real-world document Q&A applications

## Author
**Spandana M B**

## Future Improvements
 Upload PDF directly from UI
 Show source citations
 Chat history & memory
