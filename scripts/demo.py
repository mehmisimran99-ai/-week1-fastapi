#!/usr/bin/env python
"""Demo script showing all major features"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.runner import AgentRunner
from app.agent.state import AgentConfig
from app.rag.pipeline import RAGPipeline
from app.config import config


async def demo_rag():
    """Demonstrate RAG pipeline"""
    print("\n" + "="*60)
    print("RAG PIPELINE DEMO")
    print("="*60)
    
    # Create RAG pipeline
    rag = RAGPipeline(config.rag)
    
    # Ingest documents
    docs = [
        "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        "Ollama lets you run large language models locally on your machine. It supports Llama 3.1, Mistral, and more.",
        "ChromaDB is a vector database for AI applications. It stores embeddings and enables similarity search.",
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. It provides a graph-based approach to agent workflows."
    ]
    
    print("Ingesting documents...")
    count = rag.ingest(docs)
    print(f"  ✓ Ingested {count} chunks")
    
    # Query
    questions = [
        "What is FastAPI?",
        "How do you run LLMs locally?",
        "What is ChromaDB used for?",
        "What does LangGraph do?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = rag.query(q)
        print(f"A: {response.answer}")
        print(f"  Sources: {len(response.sources)}")


async def demo_agent():
    """Demonstrate agent with tools"""
    print("\n" + "="*60)
    print("AGENT DEMO")
    print("="*60)
    
    agent = AgentRunner(AgentConfig(
        model="llama3.1:8b",
        temperature=0.1,
        max_iterations=3,
    ))
    
    tasks = [
        "Calculate 15 * 23 + 100",
        "What is the square root of 144?",
        "Write Python code to compute the first 10 Fibonacci numbers",
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        response = await agent.run(task)
        print(f"Response: {response[:300]}...")


async def main():
    print("LLM ENGINEERING PORTFOLIO - FEATURE DEMO")
    print("="*60)
    
    # Check Ollama
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{config.rag.ollama_base_url}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                print("❌ Ollama not running. Start with: ollama serve")
                return
    except Exception:
        print("❌ Cannot connect to Ollama. Start with: ollama serve")
        return
    
    print("✓ Ollama connected")
    
    # Run demos
    await demo_rag()
    await demo_agent()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())