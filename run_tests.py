#!/usr/bin/env python3
"""
Space Telemetry Operations Test Suite Runner

Comprehensive test execution script for all data temperature paths,
space commands, and integration scenarios. Provides detailed reporting
and performance metrics for the complete space telemetry system.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import asyncio
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class TestResult:
    """Test execution result"""
    test_module: str
    test_function: str
    status: str  # "PASSED", "FAILED", "SKIPPED", "ERROR"
    duration_seconds: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None

@dataclass
class TestSuiteResult:
    """Complete test suite result"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    test_results: List[TestResult]
    performance_summary: Dict[str, Any]

class TestSuiteRunner:
    """Space telemetry test suite runner"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_root = project_root / "tests"
        self.results_dir = project_root / "test_results"
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Test suite configuration
        self.test_suites = {
            "hot_path": {
                "name": "HOT Path Tests",
                "module": "tests.data_paths.test_hot_path",
                "description": "Redis real-time telemetry processing (<1ms performance)",
                "critical": True
            },
            "warm_path": {
                "name": "Warm Path Tests", 
                "module": "tests.data_paths.test_warm_path",
                "description": "PostgreSQL operational analytics (<50ms queries)",
                "critical": True
            },
            "cold_path": {
                "name": "Cold Path Tests",
                "module": "tests.data_paths.test_cold_path", 
                "description": "MinIO long-term archival storage (durability & retrieval)",
                "critical": False
            },
            "analytics_path": {
                "name": "Analytics Path Tests",
                "module": "tests.data_paths.test_analytics_path",
                "description": "Vector DB & ML pipeline (anomaly detection, predictions)",
                "critical": False
            },
            "space_commands": {
                "name": "Space Command Tests",
                "module": "tests.commands.test_space_commands",
                "description": "Mission-critical commands (ABORT, SAFE_MODE, etc.)",
                "critical": True
            },
            "integration": {
                "name": "End-to-End Integration Tests",
                "module": "tests.integration.test_end_to_end", 
                "description": "Complete system integration scenarios",
                "critical": True
            }
        }
        
        self.performance_targets = {
            "hot_path_latency_ms": 1,
            "warm_path_query_ms": 50,
            "cold_path_retrieval_s": 5,
            "analytics_prediction_ms": 100,
            "command_execution_ms": 1000,
            "integration_throughput_ops": 100
        }
    
    def run_all_tests(self, parallel: bool = True, include_integration: bool = True) -> Dict[str, TestSuiteResult]:
        """Run all test suites"""
        print("🚀 Space Telemetry Operations - Test Suite Execution")
        print("=" * 60)
        print(f"Project Root: {self.project_root}")
        print(f"Test Root: {self.test_root}")
        print(f"Results Directory: {self.results_dir}")
        print(f"Parallel Execution: {parallel}")
        print(f"Include Integration: {include_integration}")
        print()
        
        # Filter test suites
        suites_to_run = {}
        for suite_id, suite_config in self.test_suites.items():
            if not include_integration and suite_id == "integration":
                print(f"⏭️  Skipping {suite_config['name']} (integration disabled)")
                continue
            suites_to_run[suite_id] = suite_config
        
        start_time = datetime.now()
        suite_results = {}
        
        if parallel and len(suites_to_run) > 1:
            # Run test suites in parallel (except integration which should run last)
            parallel_suites = {k: v for k, v in suites_to_run.items() if k != "integration"}
            integration_suite = {k: v for k, v in suites_to_run.items() if k == "integration"}
            
            if parallel_suites:
                parallel_results = self._run_suites_parallel(parallel_suites)
                suite_results.update(parallel_results)
            
            # Run integration tests last (they need other systems ready)
            if integration_suite:
                print("\n🔗 Running Integration Tests (sequential after other suites)")
                integration_results = self._run_suites_sequential(integration_suite)
                suite_results.update(integration_results)
        else:
            # Run all suites sequentially
            suite_results = self._run_suites_sequential(suites_to_run)
        
        end_time = datetime.now()
        
        # Generate comprehensive report
        self._generate_comprehensive_report(suite_results, start_time, end_time)
        
        return suite_results
    
    def _run_suites_parallel(self, suites: Dict[str, Dict]) -> Dict[str, TestSuiteResult]:
        """Run test suites in parallel"""
        print(f"⚡ Running {len(suites)} test suites in parallel...")
        
        suite_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all suite executions
            future_to_suite = {}
            for suite_id, suite_config in suites.items():
                future = executor.submit(self._run_single_suite, suite_id, suite_config)
                future_to_suite[future] = suite_id
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_suite):
                suite_id = future_to_suite[future]
                try:
                    result = future.result()
                    suite_results[suite_id] = result
                    self._print_suite_summary(suite_id, result)
                except Exception as e:
                    print(f"❌ Suite {suite_id} failed with exception: {e}")
                    traceback.print_exc()
        
        return suite_results
    
    def _run_suites_sequential(self, suites: Dict[str, Dict]) -> Dict[str, TestSuiteResult]:
        """Run test suites sequentially"""
        print(f"🔄 Running {len(suites)} test suites sequentially...")
        
        suite_results = {}
        
        for suite_id, suite_config in suites.items():
            try:
                result = self._run_single_suite(suite_id, suite_config)
                suite_results[suite_id] = result
                self._print_suite_summary(suite_id, result)
            except Exception as e:
                print(f"❌ Suite {suite_id} failed with exception: {e}")
                traceback.print_exc()
        
        return suite_results
    
    def _run_single_suite(self, suite_id: str, suite_config: Dict) -> TestSuiteResult:
        """Run a single test suite"""
        suite_name = suite_config["name"]
        module_path = suite_config["module"]
        
        print(f"\n🧪 Executing: {suite_name}")
        print(f"   Description: {suite_config['description']}")
        print(f"   Module: {module_path}")
        print(f"   Critical: {'Yes' if suite_config.get('critical', False) else 'No'}")
        
        start_time = datetime.now()
        
        # Construct pytest command
        test_file_path = self.test_root / suite_id.replace("_path", "_path") / f"test_{suite_id}.py"
        if suite_id == "space_commands":
            test_file_path = self.test_root / "commands" / "test_space_commands.py"
        elif suite_id == "integration":
            test_file_path = self.test_root / "integration" / "test_end_to_end.py"
        
        # Ensure test file exists
        if not test_file_path.exists():
            print(f"❌ Test file not found: {test_file_path}")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=0,
                test_results=[],
                performance_summary={}
            )
        
        # Run pytest with JSON output
        json_output_file = self.results_dir / f"{suite_id}_results.json"
        
        pytest_cmd = [
            sys.executable, "-m", "pytest",
            str(test_file_path),
            "--json-report", 
            f"--json-report-file={json_output_file}",
            "--asyncio-mode=auto",
            "-v",
            "--tb=short"
        ]
        
        try:
            # Execute pytest
            process_start = time.perf_counter()
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per suite
            )
            process_duration = time.perf_counter() - process_start
            
            end_time = datetime.now()
            
            # Parse pytest JSON output
            suite_result = self._parse_pytest_results(
                json_output_file, suite_name, start_time, end_time, process_duration
            )
            
            # Print test output for debugging
            if result.stdout:
                print("📄 Test Output:")
                print(result.stdout[-1000:])  # Last 1000 chars
            
            if result.stderr:
                print("⚠️  Error Output:")
                print(result.stderr[-500:])   # Last 500 chars
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Suite {suite_name} timed out after 10 minutes")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=600,
                test_results=[],
                performance_summary={"error": "timeout"}
            )
        
        except Exception as e:
            print(f"❌ Error running suite {suite_name}: {e}")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=datetime.now(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=0,
                test_results=[],
                performance_summary={"error": str(e)}
            )
    
    def _parse_pytest_results(self, json_file: Path, suite_name: str, 
                            start_time: datetime, end_time: datetime,
                            duration: float) -> TestSuiteResult:
        """Parse pytest JSON results"""
        try:
            if json_file.exists():
                with open(json_file, 'r') as f:
                    pytest_data = json.load(f)
                
                # Extract test results
                test_results = []
                passed = failed = skipped = error = 0
                
                for test in pytest_data.get("tests", []):
                    outcome = test.get("outcome", "unknown")
                    
                    if outcome == "passed":
                        passed += 1
                        status = "PASSED"
                    elif outcome == "failed":
                        failed += 1
                        status = "FAILED"
                    elif outcome == "skipped":
                        skipped += 1
                        status = "SKIPPED"
                    else:
                        error += 1
                        status = "ERROR"
                    
                    test_result = TestResult(
                        test_module=test.get("nodeid", "").split("::")[0],
                        test_function=test.get("nodeid", "").split("::")[-1],
                        status=status,
                        duration_seconds=test.get("duration", 0),
                        error_message=test.get("call", {}).get("longrepr") if outcome == "failed" else None
                    )
                    test_results.append(test_result)
                
                total_tests = len(test_results)
                
                # Extract performance summary from pytest summary
                summary = pytest_data.get("summary", {})
                performance_summary = {
                    "total_duration": duration,
                    "pytest_summary": summary
                }
                
            else:
                # Fallback if JSON file doesn't exist
                passed = failed = skipped = error = total_tests = 0
                test_results = []
                performance_summary = {"error": "no_json_output"}
            
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                total_tests=total_tests,
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                error_tests=error,
                total_duration=duration,
                test_results=test_results,
                performance_summary=performance_summary
            )
            
        except Exception as e:
            print(f"❌ Error parsing results for {suite_name}: {e}")
            return TestSuiteResult(
                suite_name=suite_name,
                start_time=start_time,
                end_time=end_time,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration=duration,
                test_results=[],
                performance_summary={"parse_error": str(e)}
            )
    
    def _print_suite_summary(self, suite_id: str, result: TestSuiteResult):
        """Print summary for a single test suite"""
        status_emoji = "✅" if result.failed_tests == 0 and result.error_tests == 0 else "❌"
        critical_marker = "🔴" if self.test_suites[suite_id].get("critical", False) else "🟢"
        
        print(f"\n{status_emoji} {critical_marker} {result.suite_name}")
        print(f"   Duration: {result.total_duration:.2f}s")
        print(f"   Tests: {result.total_tests} total, {result.passed_tests} passed, {result.failed_tests} failed")
        
        if result.failed_tests > 0 or result.error_tests > 0:
            print(f"   ❌ Failed: {result.failed_tests}, Errors: {result.error_tests}")
            
            # Show first few failures
            failures = [t for t in result.test_results if t.status in ["FAILED", "ERROR"]]
            for failure in failures[:3]:  # Show first 3 failures
                print(f"      • {failure.test_function}: {failure.status}")
                if failure.error_message:
                    error_preview = failure.error_message[:100] + "..." if len(failure.error_message) > 100 else failure.error_message
                    print(f"        {error_preview}")
    
    def _generate_comprehensive_report(self, suite_results: Dict[str, TestSuiteResult],
                                     start_time: datetime, end_time: datetime):
        """Generate comprehensive test report"""
        total_duration = (end_time - start_time).total_seconds()
        
        print(f"\n🏁 SPACE TELEMETRY TEST SUITE - FINAL REPORT")
        print("=" * 70)
        print(f"Execution Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print()
        
        # Overall statistics
        total_tests = sum(r.total_tests for r in suite_results.values())
        total_passed = sum(r.passed_tests for r in suite_results.values())
        total_failed = sum(r.failed_tests for r in suite_results.values())
        total_errors = sum(r.error_tests for r in suite_results.values())
        total_skipped = sum(r.skipped_tests for r in suite_results.values())
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 OVERALL STATISTICS")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "   Passed: 0")
        print(f"   Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "   Failed: 0")
        print(f"   Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)" if total_tests > 0 else "   Errors: 0")
        print(f"   Skipped: {total_skipped} ({total_skipped/total_tests*100:.1f}%)" if total_tests > 0 else "   Skipped: 0")
        print(f"   Success Rate: {success_rate:.1f}%")
        print()
        
        # Critical systems status
        print(f"🔴 CRITICAL SYSTEMS STATUS")
        critical_failures = 0
        for suite_id, result in suite_results.items():
            if self.test_suites[suite_id].get("critical", False):
                status = "✅ PASS" if result.failed_tests == 0 and result.error_tests == 0 else "❌ FAIL"
                if "FAIL" in status:
                    critical_failures += 1
                
                print(f"   {self.test_suites[suite_id]['name']}: {status}")
                if result.failed_tests > 0 or result.error_tests > 0:
                    print(f"     Failures: {result.failed_tests}, Errors: {result.error_tests}")
        
        critical_status = "✅ ALL CRITICAL SYSTEMS OPERATIONAL" if critical_failures == 0 else f"❌ {critical_failures} CRITICAL SYSTEM FAILURES"
        print(f"\n   Overall Critical Status: {critical_status}")
        print()
        
        # Performance summary
        print(f"⚡ PERFORMANCE SUMMARY")
        for suite_id, result in suite_results.items():
            suite_name = self.test_suites[suite_id]["name"]
            avg_test_duration = result.total_duration / result.total_tests if result.total_tests > 0 else 0
            
            print(f"   {suite_name}:")
            print(f"     Total Duration: {result.total_duration:.2f}s")
            print(f"     Avg Test Duration: {avg_test_duration:.3f}s")
            print(f"     Tests/Second: {result.total_tests/result.total_duration:.1f}" if result.total_duration > 0 else "     Tests/Second: N/A")
        print()
        
        # Data path performance
        print(f"📈 DATA PATH PERFORMANCE")
        path_performance = {
            "HOT Path (Redis)": "Real-time telemetry processing - Target: <1ms",
            "Warm Path (PostgreSQL)": "Operational analytics - Target: <50ms",
            "Cold Path (MinIO)": "Long-term archival - Target: <5s retrieval",
            "Analytics Path (ML/Vector DB)": "Predictive analytics - Target: <100ms"
        }
        
        for path_name, description in path_performance.items():
            # Find corresponding test results
            path_key = path_name.lower().split()[0] + "_path"
            if path_key in suite_results:
                result = suite_results[path_key]
                status = "✅" if result.failed_tests == 0 and result.error_tests == 0 else "❌"
                print(f"   {status} {path_name}: {description}")
                print(f"     Tests: {result.passed_tests}/{result.total_tests} passed")
            else:
                print(f"   ⏭️  {path_name}: {description} (not tested)")
        print()
        
        # Mission readiness assessment
        mission_readiness_score = self._calculate_mission_readiness(suite_results)
        
        print(f"🚀 MISSION READINESS ASSESSMENT")
        print(f"   Overall Readiness Score: {mission_readiness_score:.1f}%")
        
        if mission_readiness_score >= 95:
            readiness_status = "🟢 MISSION READY - All systems operational"
        elif mission_readiness_score >= 85:
            readiness_status = "🟡 MISSION CONDITIONAL - Minor issues detected"
        elif mission_readiness_score >= 70:
            readiness_status = "🟠 MISSION DEGRADED - Significant issues require attention"
        else:
            readiness_status = "🔴 MISSION NOT READY - Critical failures detected"
        
        print(f"   Status: {readiness_status}")
        print()
        
        # Save detailed report to file
        report_file = self.results_dir / f"test_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        detailed_report = {
            "execution_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration": total_duration,
                "total_tests": total_tests,
                "success_rate": success_rate,
                "mission_readiness_score": mission_readiness_score
            },
            "suite_results": {
                suite_id: {
                    "name": result.suite_name,
                    "total_tests": result.total_tests,
                    "passed": result.passed_tests,
                    "failed": result.failed_tests,
                    "errors": result.error_tests,
                    "skipped": result.skipped_tests,
                    "duration": result.total_duration,
                    "critical": self.test_suites[suite_id].get("critical", False)
                }
                for suite_id, result in suite_results.items()
            },
            "critical_systems_status": critical_failures == 0,
            "performance_targets": self.performance_targets
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"📄 Detailed report saved: {report_file}")
        
        # Final status
        if critical_failures == 0 and success_rate >= 95:
            print(f"\n🎉 SPACE TELEMETRY SYSTEM - ALL SYSTEMS GO! 🚀")
        elif critical_failures == 0:
            print(f"\n⚠️  SPACE TELEMETRY SYSTEM - OPERATIONAL WITH WARNINGS ⚠️")
        else:
            print(f"\n🚨 SPACE TELEMETRY SYSTEM - CRITICAL ISSUES DETECTED 🚨")
            print(f"❌ {critical_failures} critical system(s) failed - Mission not ready")
    
    def _calculate_mission_readiness(self, suite_results: Dict[str, TestSuiteResult]) -> float:
        """Calculate mission readiness score based on test results"""
        total_score = 0
        max_score = 0
        
        for suite_id, result in suite_results.items():
            suite_config = self.test_suites[suite_id]
            
            # Critical systems have higher weight
            weight = 3 if suite_config.get("critical", False) else 1
            
            # Calculate suite score
            if result.total_tests > 0:
                suite_success_rate = result.passed_tests / result.total_tests
                suite_score = suite_success_rate * weight * 100
            else:
                suite_score = 0
            
            total_score += suite_score
            max_score += weight * 100
        
        return (total_score / max_score) if max_score > 0 else 0

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(
        description="Space Telemetry Operations Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all test suites
  python run_tests.py --suite hot_path         # Run only HOT path tests
  python run_tests.py --no-parallel            # Run sequentially
  python run_tests.py --no-integration         # Skip integration tests
  python run_tests.py --critical-only          # Run only critical system tests
        """
    )
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run all test suites (default)"
    )
    
    parser.add_argument(
        "--suite", choices=["hot_path", "warm_path", "cold_path", "analytics_path", "space_commands", "integration"],
        help="Run specific test suite only"
    )
    
    parser.add_argument(
        "--critical-only", action="store_true",
        help="Run only critical system tests"
    )
    
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Run test suites sequentially instead of in parallel"
    )
    
    parser.add_argument(
        "--no-integration", action="store_true",
        help="Skip integration tests"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="test_results",
        help="Output directory for test results (default: test_results)"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    project_root = Path(__file__).parent.parent
    runner = TestSuiteRunner(project_root)
    
    # Override results directory if specified
    if args.output_dir:
        runner.results_dir = project_root / args.output_dir
        runner.results_dir.mkdir(exist_ok=True)
    
    try:
        if args.suite:
            # Run specific suite only
            if args.suite in runner.test_suites:
                suite_config = runner.test_suites[args.suite]
                result = runner._run_single_suite(args.suite, suite_config)
                runner._print_suite_summary(args.suite, result)
                
                # Generate simple report for single suite
                success = result.failed_tests == 0 and result.error_tests == 0
                print(f"\n{'✅' if success else '❌'} {suite_config['name']} - {'PASSED' if success else 'FAILED'}")
                sys.exit(0 if success else 1)
            else:
                print(f"❌ Unknown test suite: {args.suite}")
                sys.exit(1)
        
        elif args.critical_only:
            # Run only critical system tests
            critical_suites = {
                suite_id: config for suite_id, config in runner.test_suites.items()
                if config.get("critical", False)
            }
            
            if not args.no_integration:
                # Include integration if not disabled
                if "integration" in runner.test_suites:
                    critical_suites["integration"] = runner.test_suites["integration"]
            
            print(f"🔴 Running {len(critical_suites)} critical system test suites")
            
            if critical_suites:
                suite_results = {}
                for suite_id, suite_config in critical_suites.items():
                    result = runner._run_single_suite(suite_id, suite_config)
                    suite_results[suite_id] = result
                    runner._print_suite_summary(suite_id, result)
                
                # Generate report for critical suites
                runner._generate_comprehensive_report(
                    suite_results, datetime.now(), datetime.now()
                )
                
                # Exit with failure if any critical tests failed
                critical_failures = sum(1 for r in suite_results.values() if r.failed_tests > 0 or r.error_tests > 0)
                sys.exit(0 if critical_failures == 0 else 1)
            else:
                print("❌ No critical test suites found")
                sys.exit(1)
        
        else:
            # Run all tests (default)
            suite_results = runner.run_all_tests(
                parallel=not args.no_parallel,
                include_integration=not args.no_integration
            )
            
            # Determine exit code based on critical system failures
            critical_failures = 0
            for suite_id, result in suite_results.items():
                if runner.test_suites[suite_id].get("critical", False):
                    if result.failed_tests > 0 or result.error_tests > 0:
                        critical_failures += 1
            
            sys.exit(0 if critical_failures == 0 else 1)
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
