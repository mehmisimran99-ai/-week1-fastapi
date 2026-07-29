"""Custom OpenAPI schema with examples"""
from fastapi.openapi.utils import get_openapi


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add examples
    openapi_schema["components"]["examples"] = {
        "agent_chat_request": {
            "summary": "Agent chat request",
            "value": {
                "message": "Calculate 15 * 23 + 100",
                "history": []
            }
        },
        "rag_ingest_request": {
            "summary": "RAG ingest request",
            "value": {
                "texts": [
                    "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
                    "Ollama lets you run LLMs locally on your machine."
                ],
                "metadatas": [
                    {"source": "fastapi_docs", "topic": "intro"},
                    {"source": "ollama_docs", "topic": "intro"}
                ]
            }
        },
        "rag_query_request": {
            "summary": "RAG query request",
            "value": {
                "question": "What is FastAPI?"
            }
        },
        "eval_run_request": {
            "summary": "Evaluation run request",
            "value": {
                "suite": "math",
                "test_ids": ["math_1", "math_2", "math_3"]
            }
        }
    }
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add tags metadata
    openapi_schema["tags"] = [
        {"name": "RAG", "description": "Retrieval-Augmented Generation endpoints"},
        {"name": "Agent", "description": "Tool-calling agent endpoints"},
        {"name": "Eval", "description": "Evaluation and testing endpoints"},
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema