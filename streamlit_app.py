"""Streamlit entry point for the Data Assistant.

Run with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from app.agent.explainer import StepExplainer
from app.agent.factory import build_sql_agent
from app.agent.tokens import TokenBudget
from app.config import Settings, get_settings
from app.data_loader import load_excel_to_sqlite
from app.db.pool import PostgresPool
from app.db.repository import InteractionRepository
from app.db.schema import initialize_schema
from app.logging_setup import configure_logging
from app.prompts import WELCOME_MESSAGE
from app.ui.chat import render_chat_history, render_new_turn
from app.ui.session import (
    ensure_session_state,
    get_participant_id,
    get_treatment,
    reset_session_state,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Initialising data assistant...")
def _bootstrap() -> tuple[Any, InteractionRepository, StepExplainer, TokenBudget]:
    """Construct singletons that should survive across Streamlit reruns.

    ``@st.cache_resource`` guarantees this runs once per server process.
    """
    settings: Settings = get_settings()

    pool = PostgresPool(
        dsn=settings.database_url,
        minconn=settings.pg_pool_min,
        maxconn=settings.pg_pool_max,
    )
    initialize_schema(pool, settings.interactions_table)
    repository = InteractionRepository(pool, settings.interactions_table)

    database = load_excel_to_sqlite(
        excel_path=settings.excel_file_path,
        sqlite_path=settings.sqlite_db_path,
    )

    agent = build_sql_agent(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        db=database,
    )
    explainer = StepExplainer(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        max_tokens=settings.explanation_max_tokens,
        temperature=settings.explanation_temperature,
    )
    token_budget = TokenBudget(
        model_name=settings.openai_model,
        max_total_tokens=settings.max_total_tokens,
        reserved_tokens=settings.reserved_tokens,
    )

    return agent, repository, explainer, token_budget


def main() -> None:
    """Application entry point invoked by ``streamlit run``."""
    configure_logging()

    st.set_page_config(page_title="Data Assistant")
    st.title("LLM-based Data Assistant")

    try:
        agent, repository, explainer, token_budget = _bootstrap()
    except Exception as exc:
        logger.exception("Failed to initialise data assistant")
        st.error(
            f"Failed to start the data assistant: {exc}. "
            "Check the environment variables and the database connection."
        )
        st.stop()
        return  # for the type checker

    ensure_session_state(WELCOME_MESSAGE)

    if st.sidebar.button("Clear chat history"):
        reset_session_state(WELCOME_MESSAGE)
        st.rerun()

    treatment = get_treatment()
    participant_id = get_participant_id()

    render_chat_history(treatment=treatment, repository=repository)

    user_query = st.chat_input(placeholder="Ask me anything from the database!")
    if user_query:
        render_new_turn(
            user_query=user_query,
            treatment=treatment,
            participant_id=participant_id,
            agent=agent,
            explainer=explainer,
            repository=repository,
            token_budget=token_budget,
        )


if __name__ == "__main__":
    main()
