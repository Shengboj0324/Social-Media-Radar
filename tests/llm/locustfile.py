"""Locust load testing for LLM API endpoints.

Run with:
    locust -f tests/llm/locustfile.py --host=http://localhost:8000

Web UI will be available at http://localhost:8089
"""

import random

from locust import HttpUser, between, task


class LLMUser(HttpUser):
    """Simulated user making LLM requests."""

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    # Sample prompts
    prompts = [
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

    def on_start(self):
        """Called when a simulated user starts."""
        # Login or setup if needed
        pass

    @task(10)
    def generate_simple(self):
        """Test simple generation endpoint (most common)."""
        prompt = random.choice(self.prompts)

        with self.client.post(
            "/api/llm/generate",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "content" in data and len(data["content"]) > 0:
                    response.success()
                else:
                    response.failure("Empty response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def generate_chat(self):
        """Test chat generation endpoint."""
        with self.client.post(
            "/api/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": random.choice(self.prompts)},
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def cost_optimized_generation(self):
        """Test cost-optimized routing."""
        with self.client.post(
            "/api/llm/generate",
            json={
                "prompt": random.choice(self.prompts),
                "max_tokens": 100,
                "strategy": "cost_optimized",
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def quality_optimized_generation(self):
        """Test quality-optimized routing."""
        with self.client.post(
            "/api/llm/generate",
            json={
                "prompt": random.choice(self.prompts),
                "max_tokens": 150,
                "strategy": "quality_optimized",
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/api/llm/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def get_stats(self):
        """Test statistics endpoint."""
        with self.client.get("/api/llm/stats", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "total_requests" in data:
                    response.success()
                else:
                    response.failure("Invalid stats format")
            else:
                response.failure(f"Status code: {response.status_code}")


class SpikeUser(HttpUser):
    """User for spike testing - makes rapid requests."""

    wait_time = between(0.1, 0.5)  # Very short wait time

    prompts = LLMUser.prompts

    @task
    def rapid_requests(self):
        """Make rapid requests to test spike handling."""
        prompt = random.choice(self.prompts)

        self.client.post(
            "/api/llm/generate",
            json={
                "prompt": prompt,
                "max_tokens": 50,
            },
        )

