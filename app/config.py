"""Application configuration loaded from environment variables.

All runtime knobs live here so that the rest of the codebase never reads
``os.environ`` directly. This keeps configuration explicit, testable, and
documented in a single place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables (or a ``.env`` file in the
    project root). Field defaults reflect the behaviour of the legacy script.
    """

    # ---- OpenAI ----------------------------------------------------------
    openai_api_key: SecretStr = Field(..., description="OpenAI API key.")
    openai_model: str = Field(
        default="gpt-5.4-mini",
        description="Model used for both the SQL agent and the step explainer.",
    )

    # ---- PostgreSQL ------------------------------------------------------
    database_url: str = Field(..., description="PostgreSQL connection URL.")
    interactions_table: str = Field(
        default="interactions_experiment",
        description="Name of the table that stores the per-turn interaction log.",
    )
    pg_pool_min: int = Field(default=1, ge=1)
    pg_pool_max: int = Field(default=5, ge=1)

    # ---- Data loading ----------------------------------------------------
    excel_file_path: Path = Field(
        default=Path("streamlit_agent/(US)Sample-Superstore.xlsx"),
        description="Source Excel workbook that backs the SQL agent.",
    )
    sqlite_db_path: Path = Field(
        default=Path("database.db"),
        description="On-disk SQLite database produced from the Excel workbook.",
    )

    # ---- Token budget for the agent's chat history -----------------------
    max_total_tokens: int = Field(default=14_600, ge=1)
    reserved_tokens: int = Field(default=1_000, ge=0)

    # ---- Explainer LLM call ---------------------------------------------
    explanation_max_tokens: int = Field(default=1_000, ge=1)
    explanation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # ---- LLM-as-a-judge (uncertainty estimator) --------------------------
    judge_mode: Literal["off", "geval", "mtbench", "combined"] = Field(
        default="off",
        description=(
            "How answers are graded: 'geval' (probability-weighted form "
            "filling), 'mtbench' (reference-guided single-answer grading), "
            "'combined' (both), or 'off' for no uncertainty estimate."
        ),
    )
    judge_model: str | None = Field(
        default=None,
        description="Model used by the judge; defaults to ``openai_model``.",
    )
    judge_samples: int = Field(
        default=20,
        ge=1,
        description="Samples drawn to estimate score probabilities without logprobs.",
    )
    judge_max_tokens: int = Field(default=600, ge=1)
    judge_low_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Below this 0..1 confidence: 'Low'."
    )
    judge_high_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="At or above this 0..1 confidence: 'High'."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton ``Settings`` instance."""
    return Settings()
