"""Agent state and configuration"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State for the agent graph"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    next: str = ""
    iterations: int = 0
    max_iterations: int = 5


class ToolCall(BaseModel):
    """Represents a tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    id: str


class AgentConfig(BaseModel):
    """Configuration for the agent"""
    system_prompt: str = """You are a helpful assistant with access to tools. 
Use tools when needed to answer questions accurately.
Think step by step and use tools to gather information."""
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    max_tokens: int = 2048
    max_iterations: int = 5
    ollama_base_url: str = "http://localhost:11434"