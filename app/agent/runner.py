"""Agent runner using LangGraph for tool-calling agent"""
from typing import List, Dict, Any, AsyncGenerator
import json
import httpx

from app.agent.state import AgentConfig
from app.agent.tools import get_enabled_tools, ToolResult
from app.config import config


class AgentRunner:
    """LangGraph-based agent runner with tool calling"""
    
    def __init__(self, cfg: AgentConfig = None):
        self.cfg = cfg or AgentConfig()
        self.tools = get_enabled_tools(
            self.cfg.enabled_tools,
            rag_endpoint=f"http://{config.host}:{config.port}",
            tavily_key=None
        )
        self.tool_map = {t.definition.name: t for t in self.tools}
        self.tool_definitions = [t.definition for t in self.tools]
    
    def _build_messages(self, user_input: str, history: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Build message list for Ollama"""
        messages = [{"role": "system", "content": self.cfg.system_prompt}]
        if history:
            for msg in history:
                if msg["role"] in ("user", "assistant", "system", "tool"):
                    messages.append(msg)
        messages.append({"role": "user", "content": user_input})
        return messages
    
    async def _call_ollama(self, messages: List[Dict[str, Any]], stream: bool = False) -> Any:
        """Call Ollama API"""
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.cfg.temperature,
            }
        }
        if not stream and self.tool_definitions:
            payload["tools"] = [t.model_dump() for t in self.tool_definitions]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.cfg.ollama_url}/api/chat",
                json=payload
            )
            resp.raise_for_status()
            return resp.json() if not stream else resp
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tool calls"""
        results = []
        for call in tool_calls:
            tool_name = call["function"]["name"]
            tool_args = json.loads(call["function"]["arguments"])
            
            if tool_name not in self.tool_map:
                results.append(ToolResult(
                    name=tool_name,
                    content="",
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                ))
                continue
            
            tool = self.tool_map[tool_name]
            result = await tool.execute(**tool_args)
            results.append(result)
        return results
    
    async def run(self, user_input: str, history: List[Dict[str, Any]] = None) -> str:
        """Run the agent loop (non-streaming)"""
        messages = self._build_messages(user_input, history)
        iterations = 0
        
        while iterations < self.cfg.max_iterations:
            iterations += 1
            
            response = await self._call_ollama(messages, stream=False)
            msg = response.get("message", {})
            
            # Add assistant message to history
            messages.append(msg)
            
            # Check for tool calls
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                return msg.get("content", "")
            
            # Execute tools
            tool_results = await self._execute_tool_calls(tool_calls)
            
            # Add tool results to messages
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": result.content if result.success else f"Error: {result.error}"
                })
        
        return "Max iterations reached"
    
    async def run_stream(self, user_input: str, history: List[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Run the agent loop with streaming"""
        messages = self._build_messages(user_input, history)
        iterations = 0
        
        while iterations < self.cfg.max_iterations:
            iterations += 1
            
            # Stream from Ollama
            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": self.cfg.temperature}
            }
            if iterations == 1 and self.tool_definitions:
                payload["tools"] = [t.model_dump() for t in self.tool_definitions]
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{self.cfg.ollama_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    tool_calls = []
                    content_parts = []
                    
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})
                            
                            if "content" in msg and msg["content"]:
                                content_parts.append(msg["content"])
                                yield msg["content"]
                            
                            if "tool_calls" in msg:
                                tool_calls.extend(msg["tool_calls"])
                            
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    full_content = "".join(content_parts)
                    messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})
                    
                    if not tool_calls:
                        break  # Done
                    
                    # Execute tools
                    tool_results = await self._execute_tool_calls(tool_calls)
                    for result in tool_results:
                        messages.append({
                            "role": "tool",
                            "content": result.content if result.success else f"Error: {result.error}"
                        })


async def create_agent_runner(cfg: AgentConfig = None) -> AgentRunner:
    return AgentRunner(cfg)