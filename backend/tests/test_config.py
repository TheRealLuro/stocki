"""Settings resolution: how the loader and the API find Postgres."""

import pytest

from stocki.config import Settings

DB_ENV = [
    "STOCKI_DSN",
    "STOCKI_DB_HOST",
    "STOCKI_DB_PORT",
    "STOCKI_DB_NAME",
    "STOCKI_DB_USER",
    "STOCKI_DB_PASSWORD",
    "STOCKI_CORS_ORIGINS",
    "STOCKI_DATA_DIR",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in DB_ENV:
        monkeypatch.delenv(name, raising=False)


def test_builds_dsn_from_parts(monkeypatch):
    monkeypatch.setenv("STOCKI_DB_HOST", "postgres")
    monkeypatch.setenv("STOCKI_DB_PORT", "5432")
    monkeypatch.setenv("STOCKI_DB_NAME", "stocki")
    monkeypatch.setenv("STOCKI_DB_USER", "stocki_ro")
    monkeypatch.setenv("STOCKI_DB_PASSWORD", "secret")

    assert Settings.from_env().dsn == "postgresql://stocki_ro:secret@postgres:5432/stocki"


def test_explicit_dsn_overrides_parts(monkeypatch):
    monkeypatch.setenv("STOCKI_DB_HOST", "ignored")
    monkeypatch.setenv("STOCKI_DSN", "postgresql://u:p@elsewhere:6000/other")

    assert Settings.from_env().dsn == "postgresql://u:p@elsewhere:6000/other"


def test_defaults_point_at_local_compose(monkeypatch):
    """No env at all still works: `docker compose up -d` then import, nothing to configure."""
    settings = Settings.from_env()

    assert settings.host == "localhost"
    assert settings.port == 5434
    assert settings.database == "stocki"


def test_password_is_not_in_repr(monkeypatch):
    """Settings land in tracebacks and log lines; the password must not ride along."""
    monkeypatch.setenv("STOCKI_DB_PASSWORD", "hunter2")

    assert "hunter2" not in repr(Settings.from_env())


def test_cors_origins_parse_as_a_list(monkeypatch):
    monkeypatch.setenv("STOCKI_CORS_ORIGINS", "http://localhost:3000, http://localhost:5173")

    assert Settings.from_env().cors_origins == [
        "http://localhost:3000",
        "http://localhost:5173",
    ]


def test_cors_never_defaults_to_wildcard():
    assert "*" not in Settings.from_env().cors_origins
