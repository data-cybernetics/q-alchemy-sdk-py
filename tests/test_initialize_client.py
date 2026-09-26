import pytest

from q_alchemy.initialize import OptParams, create_client


def test_opt_params_uses_pinexq_api_key_as_fallback(monkeypatch):
    monkeypatch.delenv("Q_ALCHEMY_API_KEY", raising=False)
    monkeypatch.setenv("PINEXQ_API_KEY", "test-key")

    assert OptParams().api_key == "test-key"


def test_q_alchemy_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("Q_ALCHEMY_API_KEY", "primary-key")
    monkeypatch.setenv("PINEXQ_API_KEY", "fallback-key")

    assert OptParams().api_key == "primary-key"


def test_create_client_rejects_missing_api_key():
    with pytest.raises(ValueError, match="Q_ALCHEMY_API_KEY"):
        create_client(OptParams(api_key=None))
