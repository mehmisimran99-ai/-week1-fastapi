"""Evaluation harness for agent testing"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class TestCase:
    """A single test case for evaluation"""
    id: str
    input: str
    expected_output: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a test case"""
    test_id: str
    input: str
    actual_output: str
    expected_output: Optional[str]
    passed: bool
    tools_used: List[str]
    latency_ms: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Base evaluator class"""
    
    @staticmethod
    def exact_match(actual: str, expected: str) -> bool:
        return actual.strip() == expected.strip()
    
    @staticmethod
    def contains_match(actual: str, expected: str) -> bool:
        return expected.strip().lower() in actual.strip().lower()
    
    @staticmethod
    def llm_judge(actual: str, expected: str, criteria: str = "accuracy") -> bool:
        """Use LLM to judge if actual matches expected - implement with your LLM"""
        # Placeholder - integrate with your LLM provider
        return Evaluator.contains_match(actual, expected)


class EvaluationSuite:
    """Run evaluation suite against an agent"""
    
    def __init__(self, agent_runner, evaluator: Evaluator = None):
        self.agent = agent_runner
        self.evaluator = evaluator or Evaluator()
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
    
    def add_test(self, test: TestCase):
        self.test_cases.append(test)
    
    def add_tests(self, tests: List[TestCase]):
        self.test_cases.extend(tests)
    
    def load_from_file(self, path: str):
        """Load test cases from JSON file"""
        with open(path) as f:
            data = json.load(f)
        for item in data:
            self.add_test(TestCase(**item))
    
    async def run(self, judge_fn=None) -> List[TestResult]:
        """Run all test cases"""
        self.results = []
        
        for test in self.test_cases:
            start = datetime.now()
            try:
                output = await self.agent.run(test.input)
                latency = int((datetime.now() - start).total_seconds() * 1000)
                
                # Extract tools used (would need agent to return this)
                tools_used = []
                
                # Evaluate
                passed = False
                if test.expected_output:
                    if judge_fn:
                        passed = judge_fn(output, test.expected_output)
                    else:
                        passed = self.evaluator.contains_match(output, test.expected_output)
                else:
                    passed = True  # No expected output, just check it runs
                
                result = TestResult(
                    test_id=test.id,
                    input=test.input,
                    actual_output=output,
                    expected_output=test.expected_output,
                    passed=passed,
                    tools_used=tools_used,
                    latency_ms=latency,
                    metadata=test.metadata
                )
            except Exception as e:
                latency = int((datetime.now() - start).total_seconds() * 1000)
                result = TestResult(
                    test_id=test.id,
                    input=test.input,
                    actual_output="",
                    expected_output=test.expected_output,
                    passed=False,
                    tools_used=[],
                    latency_ms=latency,
                    error=str(e),
                    metadata=test.metadata
                )
            
            self.results.append(result)
        
        return self.results
    
    def summary(self) -> Dict[str, Any]:
        """Get evaluation summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        avg_latency = sum(r.latency_ms for r in self.results) / total if total > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_latency_ms": avg_latency,
            "timestamp": datetime.now().isoformat()
        }
    
    def to_json(self) -> str:
        """Export results as JSON"""
        data = {
            "summary": self.summary(),
            "results": [
                {
                    "test_id": r.test_id,
                    "input": r.input,
                    "actual_output": r.actual_output,
                    "expected_output": r.expected_output,
                    "passed": r.passed,
                    "tools_used": r.tools_used,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }
        return json.dumps(data, indent=2)
    
    def save_report(self, path: str):
        """Save report to file"""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    def print_report(self):
        """Print human-readable report"""
        summary = self.summary()
        print(f"\n{'='*60}")
        print("EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%} | Avg Latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"{'='*60}")
        
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"\n{status} | {r.test_id} ({r.latency_ms}ms)")
            print(f"  Input: {r.input[:100]}...")
            if r.expected_output:
                print(f"  Expected: {r.expected_output[:100]}...")
            print(f"  Actual: {r.actual_output[:100]}...")
            if r.error:
                print(f"  Error: {r.error}")


# Built-in test suites
def create_math_tests() -> List[TestCase]:
    return [
        TestCase(id="math_1", input="What is 2 + 2?", expected_output="4"),
        TestCase(id="math_2", input="Calculate sqrt(144)", expected_output="12"),
        TestCase(id="math_3", input="What is 15 * 23?", expected_output="345"),
        TestCase(id="math_4", input="Compute 2**10", expected_output="1024"),
    ]


def create_reasoning_tests() -> List[TestCase]:
    return [
        TestCase(id="reason_1", input="If all A are B, and all B are C, are all A also C?", expected_output="yes"),
        TestCase(id="reason_2", input="What is the capital of France?", expected_output="Paris"),
    ]


def create_tool_tests() -> List[TestCase]:
    return [
        TestCase(
            id="tool_calc",
            input="Calculate 25 * 4 + 100",
            expected_output="200",
            expected_tools=["calculator"]
        ),
        TestCase(
            id="tool_python",
            input="Run Python code to compute sum of first 100 numbers",
            expected_output="5050",
            expected_tools=["python_execute"]
        ),
    ]