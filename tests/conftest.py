"""Pytest configuration, including Hypothesis profiles."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow],
    database=None,
)

settings.register_profile(
    "ci",
    max_examples=500,
    deadline=10000,
    suppress_health_check=[HealthCheck.too_slow],
    database=None,
    # In CI the generated examples are derived from a fixed seed, so a failure
    # is reproducible byte for byte. Locally the "dev" profile keeps exploring
    # (and Hypothesis prints the seed needed to reproduce a failure).
    derandomize=True,
)

# Default profile for local development
settings.load_profile("dev")
