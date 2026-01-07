"""Pytest configuration and shared fixtures for FastWoe tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    temp_path = Path(tempfile.mkdtemp(prefix="fastwoe_test_"))
    yield temp_path
    # Cleanup after all tests
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically clean up temporary files after each test."""
    # tmp_path is pytest's built-in fixture that automatically cleans up
    # This fixture ensures any additional cleanup happens
    yield
    # Any additional cleanup can go here if needed


@pytest.fixture(scope="function")
def clean_environment(monkeypatch):
    """Ensure clean environment variables for each test."""
    # Store original env vars if needed
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ |= original_env
