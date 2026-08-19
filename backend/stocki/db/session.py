"""Connections, schema application, and the read-only role."""

from __future__ import annotations

import logging

import psycopg
from psycopg import sql

from ..config import Settings, get_settings
from ..errors import StockiConnectionError
from ..ingest.columns import SCHEMA_SQL_PATH

logger = logging.getLogger("stocki.db")


def connect(settings: Settings | None = None, *, timeout: int = 5) -> psycopg.Connection:
    """Open a connection, or raise an error that says how to fix the problem."""
    settings = settings or get_settings()
    try:
        return psycopg.connect(settings.dsn, connect_timeout=timeout)
    except psycopg.OperationalError as exc:
        raise StockiConnectionError(
            f"cannot reach Postgres at {settings.address} -- is `docker compose up -d` "
            f"running? ({str(exc).strip()})"
        ) from exc


def apply_schema(conn: psycopg.Connection, settings: Settings | None = None) -> None:
    """Create the table, indexes, and views, then grant the read-only role."""
    conn.execute(SCHEMA_SQL_PATH.read_text(encoding="utf-8"))
    conn.commit()
    apply_grants(conn, settings)


def apply_grants(conn: psycopg.Connection, settings: Settings | None = None) -> None:
    """Give the API a role that can only SELECT.

    Even a successful injection through the API reads. Skipped with a log line
    when the connected user is not allowed to manage roles.
    """
    settings = settings or get_settings()
    role = sql.Identifier(settings.ro_user)
    database = sql.Identifier(conn.info.dbname)

    try:
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", [settings.ro_user]
        ).fetchone()
        if not exists:
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    role, sql.Literal(settings.ro_password)
                )
            )
        for statement in (
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role),
            sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role),
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(role),
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}"
            ).format(role),
        ):
            conn.execute(statement)
        conn.commit()
    except psycopg.errors.InsufficientPrivilege:
        conn.rollback()
        logger.warning(
            "skipped creating %s: the connected user cannot manage roles", settings.ro_user
        )
