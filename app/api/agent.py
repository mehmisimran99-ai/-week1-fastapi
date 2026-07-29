"""Agent API endpoints"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sse_starlette.sse import EventSourceResponse

from app.agent.runner import AgentRunner
from app.agent.state import AgentConfig

router = APIRouter(prefix="/agent", tags=["Agent"])

_agent_runner: Optional[AgentRunner] = None


def get_agent_runner() -> AgentRunner:
    global _agent_runner
    if _agent_runner is None:
        _agent_runner = AgentRunner()
    return _agent_runner


class AgentRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = None
    config: Optional[Dict[str, Any]] = None
    stream: bool = False


class AgentResponse(BaseModel):
    response: str
    history: List[Dict[str, Any]]


@router.post("/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    """Chat with the agent (non-streaming)"""
    runner = get_agent_runner()
    
    if request.config:
        runner.cfg = AgentConfig(**request.config)
    
    response = await runner.run(request.message, request.history or [])
    
    # Build updated history
    history = (request.history or []).copy()
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response})
    
    return AgentResponse(response=response, history=history)


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentRequest):
    """Chat with the agent (streaming)"""
    runner = get_agent_runner()
    
    if request.config:
        runner.cfg = AgentConfig(**request.config)
    
    async def event_generator():
        async for chunk in runner.run_stream(request.message, request.history or []):
            yield {"data": chunk}
        yield {"data": "[DONE]"}
    
    return EventSourceResponse(event_generator())


@router.get("/tools")
async def list_tools():
    """List available tools"""
    runner = get_agent_runner()
    return {
        "tools": [
            {
                "name": t.definition.name,
                "description": t.definition.description,
                "parameters": t.definition.parameters
            }
            for t in runner.tools
        ]
    }


@router.post("/configure")
async def configure_agent(config: Dict[str, Any]):
    """Update agent configuration"""
    global _agent_runner
    _agent_runner = AgentRunner(cfg=AgentConfig(**config))
    return {"status": "configured", "config": config}