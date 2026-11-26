"""Idempotent schema creation for the interactions log.

Suited to a PoC: the interactions table changes rarely and only by gaining
nullable columns, so a single ``initialize_schema`` call on startup is simpler
than a full Alembic migration tree. DDL uses ``psycopg2.sql`` so identifiers are safely
quoted even though the table name is a trusted constant.
"""
from __future__ import annotations

import logging

from psycopg2 import sql

from .pool import PostgresPool

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {tbl} (
    session_id TEXT,
    id SERIAL,
    participant_id TEXT,
    treatment TEXT,
    user_query TEXT,
    assistant_response TEXT,
    intermediate_steps TEXT,
    simplified_intermediate_steps TEXT,
    user_query_sent_time TIMESTAMP,
    response_displayed_time TIMESTAMP,
    explanation_button_displayed_time TIMESTAMP,
    explanation_clicked_time TIMESTAMP,
    explanation_clicked BOOLEAN DEFAULT FALSE,
    explanation_displayed_time TIMESTAMP,
    PRIMARY KEY (session_id, id)
)
"""

# LLM-as-a-judge columns. They are added with ``ADD COLUMN IF NOT EXISTS`` so
# a table created before the judge existed is upgraded in place; all are
# nullable and stay NULL when the judge is off or produced no score.
_JUDGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("judge_mode", "TEXT"),
    ("judge_model", "TEXT"),
    ("judge_method", "TEXT"),
    ("judge_score", "DOUBLE PRECISION"),
    ("judge_confidence", "DOUBLE PRECISION"),
    ("judge_distribution", "TEXT"),
    ("judge_rationale", "TEXT"),
    ("judge_reference_sql", "TEXT"),
    ("judge_reference_result", "TEXT"),
)

_ADD_COLUMN_SQL = "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {type}"


def initialize_schema(pool: PostgresPool, table_name: str) -> None:
    """Create the interactions table, or add any columns it is missing."""
    logger.info("Initialising schema for table %s", table_name)
    table = sql.Identifier(table_name)
    with pool.cursor() as cur:
        cur.execute(sql.SQL(_CREATE_TABLE_SQL).format(tbl=table))
        for column, column_type in _JUDGE_COLUMNS:
            cur.execute(
                sql.SQL(_ADD_COLUMN_SQL).format(
                    tbl=table, col=sql.Identifier(column), type=sql.SQL(column_type)
                )
            )
    logger.info("Schema for table %s is ready", table_name)
