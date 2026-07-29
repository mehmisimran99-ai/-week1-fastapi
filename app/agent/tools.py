"""Tool definitions and implementations"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel
import httpx
import ast
import math
import os


class ToolDefinition(BaseModel):
    """Tool definition for OpenAI function calling format"""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    """Result of a tool execution"""
    name: str
    content: str
    success: bool = True
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for tools"""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
    
    def __repr__(self):
        return f"<Tool: {self.definition.name}>"


class CalculatorTool(BaseTool):
    """Safe calculator tool"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            }
        )
    
    async def execute(self, expression: str) -> ToolResult:
        try:
            # Safe evaluation using AST
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "divmod": divmod,
            }
            allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
            
            node = ast.parse(expression, mode="eval")
            
            def check(n):
                if isinstance(n, ast.Expression):
                    return check(n.body)
                elif isinstance(n, (ast.BinOp, ast.UnaryOp)):
                    left_ok = check(n.left) if hasattr(n, 'left') else True
                    right_ok = check(n.right) if hasattr(n, 'right') else True
                    return left_ok and right_ok
                elif isinstance(n, ast.Num) or isinstance(n, ast.Constant):
                    return True
                elif isinstance(n, ast.Name):
                    return n.id in allowed_names
                elif isinstance(n, ast.Call):
                    return n.func.id in allowed_names and all(check(arg) for arg in n.args)
                return False
            
            if not check(node):
                return ToolResult(name="calculator", content="", success=False, error="Invalid expression")
            
            result = eval(compile(node, "<string>", "eval"), {"__builtins__": {}}, allowed_names)
            return ToolResult(name="calculator", content=str(result))
        except Exception as e:
            return ToolResult(name="calculator", content="", success=False, error=str(e))


class PythonExecuteTool(BaseTool):
    """Execute Python code"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_execute",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        )
    
    async def execute(self, code: str) -> ToolResult:
        try:
            namespace = {"result": None}
            exec(code, {"__builtins__": {}}, namespace)
            result = namespace.get("result", "Code executed (no result variable)")
            return ToolResult(name="python_execute", content=str(result))
        except Exception as e:
            return ToolResult(name="python_execute", content="", success=False, error=str(e))


class FileReadTool(BaseTool):
    """Read a file"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_read",
            description="Read a file from the filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        )
    
    async def execute(self, path: str) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(name="file_read", content="", success=False, error=f"File not found: {path}")
            with open(path, "r") as f:
                content = f.read()
            return ToolResult(name="file_read", content=content)
        except Exception as e:
            return ToolResult(name="file_read", content="", success=False, error=str(e))


class FileWriteTool(BaseTool):
    """Write a file"""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        )
    
    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return ToolResult(name="file_write", content=f"Written to {path}")
        except Exception as e:
            return ToolResult(name="file_write", content="", success=False, error=str(e))


class WebSearchTool(BaseTool):
    """Web search via Tavily"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        )
    
    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        if not self.api_key:
            return ToolResult(name="web_search", content="", success=False, error="API key not configured")
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": self.api_key, "query": query, "max_results": max_results},
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                content = "\n".join([f"{r['title']}: {r['content'][:200]}..." for r in results])
                return ToolResult(name="web_search", content=content)
        except Exception as e:
            return ToolResult(name="web_search", content="", success=False, error=str(e))


def get_all_tools(rag_endpoint: str = "http://localhost:8000", tavily_key: str = None) -> List[BaseTool]:
    """Get all available tools"""
    return [
        CalculatorTool(),
        PythonExecuteTool(),
        FileReadTool(),
        FileWriteTool(),
        WebSearchTool(api_key=tavily_key) if tavily_key else None,
    ]


# Filter out None
def get_enabled_tools(enabled_names: List[str], **kwargs) -> List[BaseTool]:
    all_tools = get_all_tools(**kwargs)
    return [t for t in all_tools if t and t.definition.name in enabled_names]