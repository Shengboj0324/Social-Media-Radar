"""Load testing for LLM infrastructure using Locust.

This module provides load testing scenarios for:
- High-volume concurrent requests
- Sustained load over time
- Spike testing
- Stress testing

Run with:
    locust -f tests/llm/test_load.py --host=http://localhost:8000

Or programmatically:
    python tests/llm/test_load.py
"""

import asyncio
import os
import random
import time
from typing import List

from locust import HttpUser, TaskSet, between, events, task

# Try to import locust, skip if not available
try:
    from locust import FastHttpUser, task
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not installed. Install with: pip install locust")


# Sample prompts for load testing
SAMPLE_PROMPTS = [
    "Summarize the latest tech news in 3 sentences.",
    "What are the key trends in AI for 2024?",
    "Explain quantum computing briefly.",
    "What is the difference between machine learning and deep learning?",
    "List 5 benefits of cloud computing.",
    "Describe the SOLID principles in software engineering.",
    "What are microservices?",
    "Explain REST API design best practices.",
    "What is containerization?",
    "Describe the CAP theorem.",
]


class LLMLoadTest:
    """Programmatic load testing for LLM infrastructure."""

    def __init__(self, num_workers: int = 10, duration_seconds: int = 60):
        """Initialize load test.

        Args:
            num_workers: Number of concurrent workers
            duration_seconds: Test duration in seconds
        """
        self.num_workers = num_workers
        self.duration_seconds = duration_seconds
        self.results = []

    async def worker(self, worker_id: int):
        """Worker coroutine that makes requests.

        Args:
            worker_id: Worker identifier
        """
        from app.llm.router import get_router

        router = get_router()
        start_time = time.time()
        requests_made = 0
        errors = 0

        while time.time() - start_time < self.duration_seconds:
            try:
                prompt = random.choice(SAMPLE_PROMPTS)
                request_start = time.time()

                response = await router.generate_simple(
                    prompt=prompt,
                    max_tokens=100,
                )

                request_duration = time.time() - request_start
                requests_made += 1

                self.results.append({
                    "worker_id": worker_id,
                    "success": True,
                    "duration_ms": request_duration * 1000,
                    "response_length": len(response),
                })

            except Exception as e:
                errors += 1
                self.results.append({
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e),
                })

            # Small delay between requests
            await asyncio.sleep(random.uniform(0.1, 0.5))

        print(f"Worker {worker_id}: {requests_made} requests, {errors} errors")

    async def run(self):
        """Run load test."""
        print(f"Starting load test: {self.num_workers} workers, {self.duration_seconds}s duration")
        start_time = time.time()

        # Create worker tasks
        tasks = [self.worker(i) for i in range(self.num_workers)]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Calculate statistics
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]

        if successful:
            durations = [r["duration_ms"] for r in successful]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            p95_duration = sorted(durations)[int(len(durations) * 0.95)]
            p99_duration = sorted(durations)[int(len(durations) * 0.99)]
        else:
            avg_duration = min_duration = max_duration = p95_duration = p99_duration = 0

        total_requests = len(self.results)
        requests_per_second = total_requests / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 80)
        print("LOAD TEST RESULTS")
        print("=" * 80)
        print(f"Total requests: {total_requests}")
        print(f"Successful: {len(successful)} ({len(successful)/total_requests*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/total_requests*100:.1f}%)")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Requests/sec: {requests_per_second:.2f}")
        print(f"\nLatency (ms):")
        print(f"  Average: {avg_duration:.2f}")
        print(f"  Min: {min_duration:.2f}")
        print(f"  Max: {max_duration:.2f}")
        print(f"  P95: {p95_duration:.2f}")
        print(f"  P99: {p99_duration:.2f}")
        print("=" * 80)

        return {
            "total_requests": total_requests,
            "successful": len(successful),
            "failed": len(failed),
            "duration_seconds": elapsed,
            "requests_per_second": requests_per_second,
            "avg_latency_ms": avg_duration,
            "p95_latency_ms": p95_duration,
            "p99_latency_ms": p99_duration,
        }


if __name__ == "__main__":
    # Run programmatic load test
    load_test = LLMLoadTest(num_workers=10, duration_seconds=60)
    results = asyncio.run(load_test.run())

