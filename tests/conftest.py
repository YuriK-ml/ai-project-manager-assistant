"""
Pytest configuration and shared fixtures.
"""

import os
import socket

os.environ["TELEGRAM_BOT_TOKEN"] = "123456:offline-test-token"
os.environ["OPENAI_API_KEY"] = "offline-test-key"
os.environ["USE_PROXYAPI"] = "false"

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# The production index is a module singleton. Replace only its storage during
# import so offline tests neither open nor modify the user's persistent Chroma.
from unittest.mock import patch
with patch("langchain_community.vectorstores.chroma.Chroma"):
    import rag.index


@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path."""
    return project_root


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory."""
    return tmp_path


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_12345")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_12345")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # Reduce logging in tests



@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail immediately if a test accidentally calls an external service."""
    original_connect = socket.socket.connect
    def blocked(sock, address):
        # Windows asyncio uses a local socket pair to create its event loop.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        raise RuntimeError("External network disabled in tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    original_getaddrinfo = socket.getaddrinfo
    def resolve(host, *args, **kwargs):
        if host in {"127.0.0.1", "::1", "localhost"}:
            return original_getaddrinfo(host, *args, **kwargs)
        raise RuntimeError("External DNS disabled in tests")
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    from services.openai_client import openai_client
    monkeypatch.setattr(openai_client.client, "max_retries", 0)
    import httpx
    async def blocked_http(*args, **kwargs):
        raise RuntimeError("External HTTP disabled in tests")
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked_http)
