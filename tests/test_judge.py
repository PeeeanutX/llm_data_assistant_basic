"""Unit tests for :mod:`app.agent.judge`.

The OpenAI client is replaced by a fake that replays canned completions, so
these tests need neither network access nor an API key.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from app.agent.judge import (
    AnswerJudge,
    JudgeError,
    build_judge,
    confidence_level,
    format_conversation,
    format_sql_work,
)

SCHEMA = 'CREATE TABLE "Orders" ("Region" TEXT, "Sales" REAL)'


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeClient:
    """Stands in for ``openai.OpenAI``; ``responder`` maps a request to a response."""

    def __init__(self, responder) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responder(kwargs)


def _token(token: str, alternatives: dict[str, float] | None = None) -> SimpleNamespace:
    """One output token with the probabilities of its top alternatives."""
    alternatives = alternatives or {token: 1.0}
    return SimpleNamespace(
        token=token,
        top_logprobs=[
            SimpleNamespace(token=text, logprob=math.log(probability))
            for text, probability in alternatives.items()
        ],
    )


def _response(*texts: str, tokens: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    logprobs = SimpleNamespace(content=tokens) if tokens is not None else None
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), logprobs=logprobs)
            for text in texts
        ]
    )


def _bad_request(message: str) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.test"))
    return BadRequestError(message, response=response, body=None)


def _judge(mode: str, client: FakeClient, sqlite_path: Path = Path("unused.db"), **kwargs) -> AnswerJudge:
    return AnswerJudge(
        mode=mode, model="judge-model", schema=SCHEMA, sqlite_path=sqlite_path, client=client, **kwargs
    )


def _steps(query: str = 'SELECT SUM("Sales") FROM Orders', result: str = "[(60.0,)]") -> list:
    action = SimpleNamespace(tool="sql_db_query", tool_input={"query": query})
    return [(action, result)]


@pytest.fixture
def orders_db(tmp_path: Path) -> Path:
    path = tmp_path / "orders.db"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE Orders ("Region" TEXT, "Sales" REAL)')
        connection.executemany(
            "INSERT INTO Orders VALUES (?, ?)", [("West", 10.0), ("East", 20.0), ("West", 30.0)]
        )
    return path


def _reference_then(judgement: SimpleNamespace, reference_sql: str):
    """Responder for the reference-guided modes: SQL first, judgement second."""

    def responder(kwargs: dict[str, Any]) -> SimpleNamespace:
        if "SQL query:" in kwargs["messages"][-1]["content"][-20:]:
            return _response(reference_sql)
        return judgement

    return responder


# --------------------------------------------------------------------------- #
# G-Eval
# --------------------------------------------------------------------------- #
def test_geval_weights_each_score_by_its_token_probability() -> None:
    client = FakeClient(
        lambda _: _response("4", tokens=[_token("4", {"4": 0.6, "5": 0.3, "3": 0.1})])
    )

    verdict = _judge("geval", client).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.method == "logprobs"
    assert verdict.score == pytest.approx(4.2)  # 0.6*4 + 0.3*5 + 0.1*3
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.level == "High"
    assert verdict.distribution == pytest.approx({3: 0.1, 4: 0.6, 5: 0.3})
    assert verdict.reference_sql is None
    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["top_logprobs"] == 5  # newer models reject larger values


def test_geval_finds_the_score_after_an_echoed_form_label() -> None:
    tokens = [
        _token("Correct", {"Correct": 0.7, "5": 0.3}),
        _token("ness"),
        _token(":"),
        _token(" 4", {" 4": 0.75, " 5": 0.25}),
    ]
    client = FakeClient(lambda _: _response("Correctness: 4", tokens=tokens))

    verdict = _judge("geval", client).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.method == "logprobs"
    assert verdict.distribution == pytest.approx({4: 0.75, 5: 0.25})


def test_geval_renormalises_over_score_tokens_only() -> None:
    alternatives = {" 5": 0.3, "5": 0.15, "4": 0.45, "The": 0.08, "9": 0.02}
    client = FakeClient(lambda _: _response(" 5", tokens=[_token(" 5", alternatives)]))

    verdict = _judge("geval", client).evaluate("Total sales?", _steps(), "60 dollars")

    # "The" and the out-of-range "9" are dropped; " 5" and "5" are one score.
    assert verdict.distribution == pytest.approx({4: 0.5, 5: 0.5})
    assert verdict.score == pytest.approx(4.5)


def test_geval_estimates_probabilities_by_sampling_without_logprobs() -> None:
    def responder(kwargs: dict[str, Any]) -> SimpleNamespace:
        if "n" not in kwargs:
            return _response("4")  # the model returned no logprobs
        return _response(*(["4", "2"] * kwargs["n"])[: kwargs["n"]])

    client = FakeClient(responder)

    verdict = _judge("geval", client, samples=20).evaluate("Total sales?", _steps(), "60 dollars")

    sampling_calls = [call for call in client.calls if "n" in call]
    assert [call["n"] for call in sampling_calls] == [8, 8, 4]
    assert all(call["temperature"] == 1.0 for call in sampling_calls)
    assert verdict.method == "sampling"
    assert verdict.distribution == pytest.approx({2: 0.5, 4: 0.5})
    assert verdict.score == pytest.approx(3.0)
    assert verdict.level == "Medium"


def test_geval_samples_when_the_model_rejects_logprobs() -> None:
    def responder(kwargs: dict[str, Any]) -> SimpleNamespace:
        if "logprobs" in kwargs:
            raise _bad_request("Unsupported parameter: 'logprobs' is not supported with this model.")
        return _response(*["5"] * kwargs["n"])

    client = FakeClient(responder)
    judge = _judge("geval", client, samples=4)

    first = judge.evaluate("Total sales?", _steps(), "60 dollars")
    calls_for_first = len(client.calls)
    judge.evaluate("Total sales?", _steps(), "60 dollars")

    assert first.method == "sampling"
    assert first.score == pytest.approx(5.0)
    # The rejection is remembered: the second evaluation does not retry logprobs.
    assert calls_for_first == 2
    assert len(client.calls) == 3


def test_sampling_without_any_score_raises() -> None:
    client = FakeClient(lambda kwargs: _response(*["no idea"] * kwargs.get("n", 1)))

    with pytest.raises(JudgeError):
        _judge("geval", client, samples=3).evaluate("Total sales?", _steps(), "60 dollars")


# --------------------------------------------------------------------------- #
# MT-Bench (reference-guided single-answer grading)
# --------------------------------------------------------------------------- #
def test_mtbench_grades_against_an_independently_executed_reference(orders_db: Path) -> None:
    client = FakeClient(
        _reference_then(
            _response("The assistant's total matches the reference.\n\nRating: [[9]]"),
            reference_sql='```sql\nSELECT SUM("Sales") AS total FROM Orders;\n```',
        )
    )

    verdict = _judge("mtbench", client, orders_db).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.method == "single"
    assert verdict.score == 9.0
    assert (verdict.scale_min, verdict.scale_max) == (1, 10)
    assert verdict.confidence == pytest.approx(8 / 9)
    assert verdict.distribution is None
    assert verdict.reference_sql == 'SELECT SUM("Sales") AS total FROM Orders'
    assert verdict.reference_result == "total\n60.0"
    assert verdict.rationale.endswith("Rating: [[9]]")
    judge_prompt = client.calls[1]["messages"][-1]["content"]
    assert "[The Start of Reference Answer]" in judge_prompt
    assert "total\n60.0" in judge_prompt


def test_mtbench_reference_refuses_anything_but_a_select(orders_db: Path) -> None:
    client = FakeClient(
        _reference_then(_response("Rating: [[3]]"), reference_sql="DROP TABLE Orders")
    )

    verdict = _judge("mtbench", client, orders_db).evaluate("Delete it", _steps(), "Done")

    assert verdict.reference_result == "no read-only SELECT query was produced"
    assert "No reference result is available" in client.calls[1]["messages"][-1]["content"]
    with sqlite3.connect(orders_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM Orders").fetchone() == (3,)


def test_mtbench_tells_the_judge_when_the_database_cannot_answer(orders_db: Path) -> None:
    client = FakeClient(_reference_then(_response("Rating: [[9]]"), reference_sql="NO_QUERY"))

    verdict = _judge("mtbench", client, orders_db).evaluate(
        "How many orders were returned?", [], "The database has no returns data."
    )

    assert verdict.reference_sql is None
    assert "does not contain the data needed" in verdict.reference_result
    assert "does not contain the data needed" in client.calls[1]["messages"][-1]["content"]


def test_mtbench_reports_a_failing_reference_query_to_the_judge(orders_db: Path) -> None:
    client = FakeClient(
        _reference_then(_response("Rating: [[6]]"), reference_sql="SELECT nope FROM Orders")
    )

    verdict = _judge("mtbench", client, orders_db).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.score == 6.0
    assert "the reference query failed" in verdict.reference_result


@pytest.mark.parametrize("output", ["It looks fine.", "Rating: [[11]]", "Rating: [[0]]"])
def test_mtbench_without_a_valid_rating_raises(orders_db: Path, output: str) -> None:
    client = FakeClient(_reference_then(_response(output), reference_sql="SELECT 1"))

    with pytest.raises(JudgeError):
        _judge("mtbench", client, orders_db).evaluate("Total sales?", _steps(), "60 dollars")


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #
def test_combined_reads_the_distribution_at_the_rating_token(orders_db: Path) -> None:
    tokens = [
        _token("The"),
        _token(" top"),
        _token(" 5", {" 5": 0.9, " 3": 0.1}),  # a number in the explanation, not the rating
        _token(" rows"),
        _token(" match"),
        _token(".\n\n"),
        _token("Rating"),
        _token(":"),
        _token(" [["),
        _token("4", {"4": 0.7, "5": 0.2, "3": 0.1}),
        _token("]]"),
    ]
    text = "".join(token.token for token in tokens)
    client = FakeClient(
        _reference_then(
            _response(text, tokens=tokens),
            reference_sql='SELECT "Region", SUM("Sales") FROM Orders GROUP BY "Region"',
        )
    )

    verdict = _judge("combined", client, orders_db).evaluate("Sales by region?", _steps(), "...")

    assert verdict.method == "logprobs"
    assert verdict.distribution == pytest.approx({3: 0.1, 4: 0.7, 5: 0.2})
    assert verdict.score == pytest.approx(4.1)
    assert (verdict.scale_min, verdict.scale_max) == (1, 5)
    assert verdict.rationale == text
    assert verdict.reference_result == "Region | SUM(\"Sales\")\nEast | 20.0\nWest | 40.0"
    assert "Reference Answer:" in client.calls[1]["messages"][-1]["content"]


def test_combined_handles_a_rating_merged_into_one_token(orders_db: Path) -> None:
    tokens = [_token("Correct. Rating: "), _token("[[5", {"[[5": 0.8, "[[4": 0.2}), _token("]]")]
    text = "".join(token.token for token in tokens)
    client = FakeClient(_reference_then(_response(text, tokens=tokens), reference_sql="SELECT 1"))

    verdict = _judge("combined", client, orders_db).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.distribution == pytest.approx({4: 0.2, 5: 0.8})


# --------------------------------------------------------------------------- #
# Request adaptation, construction, helpers
# --------------------------------------------------------------------------- #
def test_token_parameter_is_swapped_when_the_model_rejects_it() -> None:
    def responder(kwargs: dict[str, Any]) -> SimpleNamespace:
        if "max_completion_tokens" in kwargs:
            raise _bad_request("Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens'.")
        return _response("5", tokens=[_token("5")])

    client = FakeClient(responder)

    verdict = _judge("geval", client).evaluate("Total sales?", _steps(), "60 dollars")

    assert verdict.score == pytest.approx(5.0)
    assert "max_tokens" in client.calls[-1]
    assert "max_completion_tokens" not in client.calls[-1]


def test_unrelated_bad_request_is_not_swallowed() -> None:
    def responder(kwargs: dict[str, Any]) -> SimpleNamespace:
        raise _bad_request("This model's maximum context length is 8192 tokens.")

    with pytest.raises(BadRequestError):
        _judge("geval", FakeClient(responder)).evaluate("Total sales?", _steps(), "60 dollars")


def test_build_judge_returns_none_when_off() -> None:
    judge = build_judge(
        mode="off", api_key="sk-test", model="judge-model", db=None, sqlite_path=Path("unused.db")
    )

    assert judge is None


def test_build_judge_reads_the_schema_from_the_database() -> None:
    database = SimpleNamespace(get_table_info=lambda: SCHEMA)

    judge = build_judge(
        mode="combined",
        api_key="sk-test",
        model="judge-model",
        db=database,
        sqlite_path=Path("unused.db"),
    )

    assert judge is not None
    assert judge.mode == "combined"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown judge mode"):
        _judge("pairwise", FakeClient(lambda _: None))


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(1.0, "High"), (0.75, "High"), (0.74, "Medium"), (0.5, "Medium"), (0.49, "Low"), (0.0, "Low")],
)
def test_confidence_level_buckets(confidence: float, expected: str) -> None:
    assert confidence_level(confidence, low_threshold=0.5, high_threshold=0.75) == expected


def test_format_sql_work_keeps_queries_and_shortens_schema_lookups() -> None:
    steps = [
        (SimpleNamespace(tool="sql_db_list_tables", tool_input={}), "Orders, People"),
        (SimpleNamespace(tool="sql_db_schema", tool_input={"table_names": "Orders"}), "CREATE ..."),
        *_steps(),
    ]

    rendered = format_sql_work(steps)

    assert "Step 1 - called sql_db_list_tables" in rendered
    assert "CREATE ..." not in rendered
    assert 'Step 3 - ran SQL query:\nSELECT SUM("Sales") FROM Orders\nResult:\n[(60.0,)]' in rendered
    assert format_sql_work([]) == "(no tool calls)"


def test_format_conversation_leaves_out_the_pending_question() -> None:
    history = [
        SimpleNamespace(type="ai", content="Hello!"),
        SimpleNamespace(type="human", content="Total sales in 2021?"),
        SimpleNamespace(type="ai", content="100 dollars."),
        SimpleNamespace(type="human", content="And in 2022?"),
    ]

    rendered = format_conversation(history, "And in 2022?")

    assert rendered == "Assistant: Hello!\nUser: Total sales in 2021?\nAssistant: 100 dollars."
    assert format_conversation([], "Hi") == "(none)"
