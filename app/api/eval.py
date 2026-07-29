"""Evaluation API endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from app.eval.harness import EvaluationSuite, TestCase, create_math_tests, create_reasoning_tests, create_tool_tests
from app.agent.runner import AgentRunner
from app.agent.state import AgentConfig

router = APIRouter(prefix="/eval", tags=["Evaluation"])

# Store evaluation runs
_eval_runs: Dict[str, Dict] = {}


class EvalRunRequest(BaseModel):
    suite: str = "full"  # math, reasoning, tools, full, custom
    custom_tests: Optional[List[TestCase]] = None
    agent_config: Optional[Dict[str, Any]] = None


class EvalRunResponse(BaseModel):
    run_id: str
    status: str
    started_at: str


class EvalResultResponse(BaseModel):
    run_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None


def _build_test_cases(request: EvalRunRequest) -> List[TestCase]:
    """Build test cases from request"""
    tests = []
    if request.suite in ("full", "math"):
        tests.extend(create_math_tests())
    if request.suite in ("full", "reasoning"):
        tests.extend(create_reasoning_tests())
    if request.suite in ("full", "tools"):
        tests.extend(create_tool_tests())
    if request.custom_tests:
        tests.extend(request.custom_tests)
    return tests


@router.post("/run", response_model=EvalRunResponse)
async def run_evaluation(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """Start an evaluation run"""
    run_id = str(uuid.uuid4())[:8]
    
    # Create agent runner
    agent = AgentRunner(AgentConfig(**(request.agent_config or {})))
    
    # Build test cases
    test_cases = _build_test_cases(request)
    if not test_cases:
        raise HTTPException(400, "No tests specified")
    
    # Create suite
    suite = EvaluationSuite(agent)
    suite.add_tests(test_cases)
    
    # Store run info
    _eval_runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "suite": request.suite,
        "test_count": len(test_cases),
        "results": None,
        "summary": None,
    }
    
    # Run in background
    background_tasks.add_task(_run_eval_background, run_id, suite)
    
    return EvalRunResponse(
        run_id=run_id,
        status="running",
        started_at=_eval_runs[run_id]["started_at"],
    )


async def _run_eval_background(run_id: str, suite: EvaluationSuite):
    """Run evaluation in background"""
    try:
        results = await suite.run()
        _eval_runs[run_id]["status"] = "completed"
        _eval_runs[run_id]["completed_at"] = datetime.now().isoformat()
        _eval_runs[run_id]["results"] = [
            {
                "test_id": r.test_id,
                "input": r.input,
                "actual_output": r.actual_output,
                "expected_output": r.expected_output,
                "passed": r.passed,
                "tools_used": r.tools_used,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in results
        ]
        _eval_runs[run_id]["summary"] = suite.summary()
    except Exception as e:
        _eval_runs[run_id]["status"] = "failed"
        _eval_runs[run_id]["error"] = str(e)


@router.get("/run/{run_id}", response_model=EvalResultResponse)
async def get_eval_result(run_id: str):
    """Get evaluation run result"""
    if run_id not in _eval_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = _eval_runs[run_id]
    return EvalResultResponse(
        run_id=run["run_id"],
        status=run["status"],
        started_at=run["started_at"],
        completed_at=run.get("completed_at"),
        summary=run.get("summary"),
        results=run.get("results"),
    )


@router.get("/runs")
async def list_eval_runs():
    """List all evaluation runs"""
    return [
        {
            "run_id": run["run_id"],
            "status": run["status"],
            "started_at": run["started_at"],
            "completed_at": run.get("completed_at"),
            "suite": run.get("suite"),
            "test_count": run.get("test_count"),
            "passed": run.get("summary", {}).get("passed") if run.get("summary") else None,
        }
        for run in _eval_runs.values()
    ]


@router.get("/suites")
async def list_suites():
    """List available test suites"""
    return {
        "suites": [
            {"name": "math", "description": "Mathematical calculation tests", "count": len(create_math_tests())},
            {"name": "reasoning", "description": "Logical reasoning tests", "count": len(create_reasoning_tests())},
            {"name": "tools", "description": "Tool usage tests", "count": len(create_tool_tests())},
            {"name": "full", "description": "All tests combined", "count": len(create_math_tests()) + len(create_reasoning_tests()) + len(create_tool_tests())},
        ]
    }