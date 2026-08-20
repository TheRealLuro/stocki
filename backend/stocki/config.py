"""Environment-driven settings.

Cason should never have to configure anything: with nothing set, the defaults
point at the Postgres that `docker compose up -d` starts on the host. Inside
the compose network the API overrides host/port through environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

# Read .env once at import so a plain `import stocki` picks up local credentials.
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5434  # 5432 and 5433 are commonly taken by other local stacks
DEFAULT_NAME = "stocki"
DEFAULT_USER = "stocki"
DEFAULT_CORS = ["http://localhost:3000", "http://localhost:5173"]


def _split_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    database: str = DEFAULT_NAME
    user: str = DEFAULT_USER
    password: str = field(default="stocki", repr=False)
    dsn_override: str | None = field(default=None, repr=False)
    cors_origins: list[str] = field(default_factory=lambda: list(DEFAULT_CORS))
    data_dir: Path = REPO_ROOT / "data"
    rate_limit_per_minute: int = 120
    dataset_rate_limit_per_minute: int = 20
    max_body_bytes: int = 1_000_000
    ro_user: str = "stocki_ro"
    ro_password: str = field(default="stocki_ro", repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        raw_origins = os.getenv("STOCKI_CORS_ORIGINS")
        raw_data_dir = os.getenv("STOCKI_DATA_DIR")
        return cls(
            ro_user=os.getenv("STOCKI_RO_USER", "stocki_ro"),
            ro_password=os.getenv("STOCKI_RO_PASSWORD", "stocki_ro"),
            rate_limit_per_minute=int(os.getenv("STOCKI_RATE_LIMIT", "120")),
            dataset_rate_limit_per_minute=int(os.getenv("STOCKI_DATASET_RATE_LIMIT", "20")),
            max_body_bytes=int(os.getenv("STOCKI_MAX_BODY_BYTES", "1000000")),
            host=os.getenv("STOCKI_DB_HOST", DEFAULT_HOST),
            port=int(os.getenv("STOCKI_DB_PORT", str(DEFAULT_PORT))),
            database=os.getenv("STOCKI_DB_NAME", DEFAULT_NAME),
            user=os.getenv("STOCKI_DB_USER", DEFAULT_USER),
            password=os.getenv("STOCKI_DB_PASSWORD", "stocki"),
            dsn_override=os.getenv("STOCKI_DSN"),
            cors_origins=_split_origins(raw_origins) if raw_origins else list(DEFAULT_CORS),
            data_dir=Path(raw_data_dir) if raw_data_dir else REPO_ROOT / "data",
        )

    @property
    def dsn(self) -> str:
        if self.dsn_override:
            return self.dsn_override
        return (
            f"postgresql://{quote(self.user)}:{quote(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def safe_dsn(self) -> str:
        """DSN with the password masked, for logs and error messages."""
        if self.dsn_override:
            return "postgresql://***@" + self.dsn_override.rsplit("@", 1)[-1]
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"

    @property
    def address(self) -> str:
        """host:port, for 'is the database up?' messages."""
        if self.dsn_override:
            return self.dsn_override.rsplit("@", 1)[-1].split("/")[0]
        return f"{self.host}:{self.port}"


def get_settings() -> Settings:
    return Settings.from_env()
