import pytest

@pytest.fixture(autouse=True)
def interservice_caller_environment(monkeypatch):
    monkeypatch.setenv("INTERSERVICE_SECRET", "test-interservice-credential")
