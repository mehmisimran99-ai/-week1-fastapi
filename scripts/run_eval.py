#!/usr/bin/env python
"""Run evaluation suite and generate report"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.eval.harness import EvaluationSuite, create_math_tests, create_reasoning_tests, create_tool_tests
from app.agent.runner import AgentRunner
from app.agent.state import AgentConfig


async def main():
    print("=" * 60)
    print("LLM ENGINEERING PORTFOLIO - EVALUATION SUITE")
    print("=" * 60)
    
    # Create agent
    agent = AgentRunner(AgentConfig(
        model="llama3.1:8b",
        temperature=0.1,
        max_iterations=5,
    ))
    
    # Build test suite
    suite = EvaluationSuite(agent)
    suite.add_tests(create_math_tests())
    suite.add_tests(create_reasoning_tests())
    suite.add_tests(create_tool_tests())
    
    print(f"\nRunning {len(suite.test_cases)} tests...")
    print("-" * 60)
    
    # Run evaluation
    await suite.run()
    
    # Print report
    suite.print_report()
    
    # Save results
    report_path = Path(__file__).parent.parent / "eval_results.json"
    suite.save_report(str(report_path))
    print(f"\nReport saved to: {report_path}")
    
    # Exit code based on pass rate
    summary = suite.summary()
    if summary["pass_rate"] >= 0.8:
        print("\n✅ EVALUATION PASSED")
        return 0
    else:
        print("\n❌ EVALUATION FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)