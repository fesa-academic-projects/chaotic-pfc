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
)

# Default profile for local development
settings.load_profile("dev")
