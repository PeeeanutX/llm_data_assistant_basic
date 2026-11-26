"""LLM-as-a-judge uncertainty estimator for the SQL agent's answers.

A third LLM pass grades every database-grounded answer; the grade is the
assistant's "how sure am I?" signal. Three interchangeable modes are offered
so they can be compared on the same questions:

``geval``
    G-Eval (Liu et al., 2023). The prompt carries the evaluation criteria and
    frozen evaluation steps, the judge fills in a form with a 1-5 score, and
    the final score is the probability-weighted sum over the score tokens. If
    the model exposes no token probabilities they are estimated by sampling
    (the paper's ``n = 20, temperature = 1``).

``mtbench``
    Reference-guided single-answer grading (Zheng et al., 2023). The judge
    model first answers the question independently (its own SQL, executed
    read-only); that reference is then shown to the judge, which explains its
    assessment and ends with ``Rating: [[n]]`` on a 1-10 scale.

``combined``
    The reference-guided judge of ``mtbench`` with the rubric and
    probability-weighted 1-5 readout of ``geval``.

Setting the mode to ``off`` disables the judge entirely (see ``build_judge``).
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

from app.prompts import (
    COMBINED_JUDGE_SYSTEM_PROMPT,
    GEVAL_JUDGE_SYSTEM_PROMPT,
    MTBENCH_JUDGE_SYSTEM_PROMPT,
    REFERENCE_NO_QUERY,
    REFERENCE_SQL_SYSTEM_PROMPT,
    build_combined_user_prompt,
    build_geval_user_prompt,
    build_mtbench_user_prompt,
    build_reference_sql_user_prompt,
)

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_GEVAL = "geval"
MODE_MTBENCH = "mtbench"
MODE_COMBINED = "combined"
JUDGE_MODES: tuple[str, ...] = (MODE_GEVAL, MODE_MTBENCH, MODE_COMBINED)

METHOD_LOGPROBS = "logprobs"
METHOD_SAMPLING = "sampling"
METHOD_SINGLE = "single"

_GEVAL_SCALE = (1, 5)
_MTBENCH_SCALE = (1, 10)

_NEW_TOKEN_PARAM = "max_completion_tokens"
_LEGACY_TOKEN_PARAM = "max_tokens"

_FORM_MAX_TOKENS = 10  # G-Eval form-filling: the score, with room for an echoed label
_TOP_LOGPROBS = 5  # one per score on the 1-5 scale; newer models reject more than 5
_MAX_CHOICES_PER_REQUEST = 8  # newer OpenAI models reject n > 8
_MAX_PARAM_ADAPTATIONS = 2
_REQUEST_TIMEOUT_SECONDS = 60.0
_SQL_TIMEOUT_SECONDS = 10.0
_MAX_REFERENCE_ROWS = 50
_MAX_BLOCK_CHARS = 3_000
_MAX_HISTORY_MESSAGES = 6
_MAX_HISTORY_CHARS = 1_000

_BRACKET_RATING = re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]")
_LABELLED_RATING = re.compile(r"Rating:\s*\[?\[?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_SCORE_TOKEN = re.compile(r"[\s\[(*:]*(\d{1,2})[\s\])*.]*")
_SQL_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class JudgeError(RuntimeError):
    """The judge could not produce a usable score."""


class _LogprobsUnsupported(Exception):
    """The model rejected the ``logprobs`` request parameters."""


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's assessment of one answer.

    ``score`` is on the mode's native scale (``scale_min``..``scale_max``);
    ``confidence`` is the same score rescaled to 0..1 and ``level`` its
    High / Medium / Low bucket. ``distribution`` holds the probability of each
    score (``None`` for ``mtbench``, which yields a single rating).
    """

    mode: str
    model: str
    method: str
    score: float
    scale_min: int
    scale_max: int
    confidence: float
    level: str
    distribution: dict[int, float] | None = None
    rationale: str | None = None
    reference_sql: str | None = None
    reference_result: str | None = None


@dataclass(frozen=True)
class _Reference:
    """The judge model's own, independently produced answer."""

    sql: str | None
    result: str | None
    problem: str | None = None

    def as_prompt_text(self) -> str:
        if self.problem is None:
            return f"SQL query:\n{self.sql}\n\nResult:\n{self.result}"
        sql_part = f"SQL query:\n{self.sql}\n\n" if self.sql else ""
        return (
            f"{sql_part}(No reference result is available: {self.problem}. "
            "Judge the assistant's answer on its own merits.)"
        )


class AnswerJudge:
    """Grades an answer of the SQL agent in one of the three judging modes."""

    def __init__(
        self,
        *,
        mode: str,
        model: str,
        schema: str,
        sqlite_path: Path,
        api_key: str | None = None,
        samples: int = 20,
        max_tokens: int = 600,
        low_threshold: float = 0.5,
        high_threshold: float = 0.75,
        client: Any | None = None,
    ) -> None:
        if mode not in JUDGE_MODES:
            raise ValueError(f"Unknown judge mode {mode!r}; expected one of {JUDGE_MODES}")
        self._mode = mode
        self._model = model
        self._schema = schema
        self._sqlite_path = sqlite_path
        self._samples = samples
        self._max_tokens = max_tokens
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._client = (
            client
            if client is not None
            else OpenAI(api_key=api_key, timeout=_REQUEST_TIMEOUT_SECONDS)
        )
        # Optimistic defaults, corrected on the first request the model rejects.
        self._token_param: str = _NEW_TOKEN_PARAM
        self._send_temperature: bool = True
        self._logprobs_supported: bool = True

    @property
    def mode(self) -> str:
        return self._mode

    def evaluate(
        self,
        question: str,
        steps: Iterable[tuple[Any, Any]],
        answer: str,
        history: Sequence[Any] = (),
    ) -> JudgeVerdict:
        """Grade ``answer`` to ``question`` given the agent's intermediate ``steps``.

        ``history`` is the chat history (LangChain messages) the agent saw; it
        gives the judge the context needed to understand follow-up questions.

        Raises:
            JudgeError: If no score could be extracted from the judge's output.
        """
        sql_work = format_sql_work(steps)
        conversation = format_conversation(history, question)

        if self._mode == MODE_GEVAL:
            return self._evaluate_geval(question, conversation, sql_work, answer)

        reference = self._build_reference(question, conversation)
        if self._mode == MODE_MTBENCH:
            return self._evaluate_mtbench(question, conversation, sql_work, answer, reference)
        return self._evaluate_combined(question, conversation, sql_work, answer, reference)

    # ------------------------------------------------------------------ #
    # Modes
    # ------------------------------------------------------------------ #
    def _evaluate_geval(
        self, question: str, conversation: str, sql_work: str, answer: str
    ) -> JudgeVerdict:
        low, high = _GEVAL_SCALE
        messages = [
            {"role": "system", "content": GEVAL_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_geval_user_prompt(
                    self._schema, conversation, question, sql_work, answer
                ),
            },
        ]
        distribution, method, _ = self._probability_weighted(
            messages,
            max_tokens=_FORM_MAX_TOKENS,
            parse_score=lambda text: _parse_form_score(text, low, high),
            locate_score_token=lambda tokens, _text: _first_score_token(tokens, low, high),
            scale=_GEVAL_SCALE,
        )
        return self._verdict(
            score=_weighted_score(distribution),
            scale=_GEVAL_SCALE,
            method=method,
            distribution=distribution,
        )

    def _evaluate_mtbench(
        self,
        question: str,
        conversation: str,
        sql_work: str,
        answer: str,
        reference: _Reference,
    ) -> JudgeVerdict:
        low, high = _MTBENCH_SCALE
        messages = [
            {"role": "system", "content": MTBENCH_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_mtbench_user_prompt(
                    self._schema,
                    conversation,
                    question,
                    reference.as_prompt_text(),
                    sql_work,
                    answer,
                ),
            },
        ]
        response = self._create(messages, max_tokens=self._max_tokens, temperature=0.0)
        text = response.choices[0].message.content or ""
        match = _find_rating(text)
        rating = float(match.group(1)) if match else None
        if rating is None or not low <= rating <= high:
            raise JudgeError(f"No rating between {low} and {high} in the judge's output")
        return self._verdict(
            score=rating,
            scale=_MTBENCH_SCALE,
            method=METHOD_SINGLE,
            rationale=text.strip(),
            reference=reference,
        )

    def _evaluate_combined(
        self,
        question: str,
        conversation: str,
        sql_work: str,
        answer: str,
        reference: _Reference,
    ) -> JudgeVerdict:
        low, high = _GEVAL_SCALE
        messages = [
            {"role": "system", "content": COMBINED_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_combined_user_prompt(
                    self._schema,
                    conversation,
                    question,
                    reference.as_prompt_text(),
                    sql_work,
                    answer,
                ),
            },
        ]
        distribution, method, rationale = self._probability_weighted(
            messages,
            max_tokens=self._max_tokens,
            parse_score=lambda text: _parse_integer_rating(text, low, high),
            locate_score_token=_rating_token,
            scale=_GEVAL_SCALE,
        )
        return self._verdict(
            score=_weighted_score(distribution),
            scale=_GEVAL_SCALE,
            method=method,
            distribution=distribution,
            rationale=rationale.strip(),
            reference=reference,
        )

    def _verdict(
        self,
        *,
        score: float,
        scale: tuple[int, int],
        method: str,
        distribution: dict[int, float] | None = None,
        rationale: str | None = None,
        reference: _Reference | None = None,
    ) -> JudgeVerdict:
        low, high = scale
        confidence = min(1.0, max(0.0, (score - low) / (high - low)))
        return JudgeVerdict(
            mode=self._mode,
            model=self._model,
            method=method,
            score=score,
            scale_min=low,
            scale_max=high,
            confidence=confidence,
            level=confidence_level(confidence, self._low_threshold, self._high_threshold),
            distribution=distribution,
            rationale=rationale or None,
            reference_sql=reference.sql if reference else None,
            reference_result=(reference.result or reference.problem) if reference else None,
        )

    # ------------------------------------------------------------------ #
    # G-Eval scoring: probability-weighted summation
    # ------------------------------------------------------------------ #
    def _probability_weighted(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        parse_score: Callable[[str], int | None],
        locate_score_token: Callable[[Sequence[Any], str], Any | None],
        scale: tuple[int, int],
    ) -> tuple[dict[int, float], str, str]:
        """Return ``(score distribution, method, judge output text)``.

        Reads the distribution off the score token's log-probabilities when
        the model exposes them, and estimates it by sampling otherwise.
        """
        if self._logprobs_supported:
            try:
                response = self._create(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=_TOP_LOGPROBS,
                )
            except _LogprobsUnsupported:
                logger.info("Model %s exposes no logprobs; sampling instead", self._model)
                self._logprobs_supported = False
            else:
                choice = response.choices[0]
                text = choice.message.content or ""
                distribution = _distribution_from_logprobs(choice, text, locate_score_token, scale)
                if distribution is not None:
                    return distribution, METHOD_LOGPROBS, text
                logger.info("No score token in the judge's logprobs; sampling instead")

        distribution, text = self._sampled_distribution(messages, max_tokens, parse_score)
        return distribution, METHOD_SAMPLING, text

    def _sampled_distribution(
        self,
        messages: list[dict],
        max_tokens: int,
        parse_score: Callable[[str], int | None],
    ) -> tuple[dict[int, float], str]:
        """Estimate the score distribution from ``samples`` completions."""
        counts: Counter[int] = Counter()
        first_scored_text = ""
        remaining = self._samples
        while remaining > 0:
            batch = min(remaining, _MAX_CHOICES_PER_REQUEST)
            response = self._create(messages, max_tokens=max_tokens, temperature=1.0, n=batch)
            for choice in response.choices:
                text = choice.message.content or ""
                score = parse_score(text)
                if score is not None:
                    counts[score] += 1
                    first_scored_text = first_scored_text or text
            remaining -= batch

        total = sum(counts.values())
        if total == 0:
            raise JudgeError("None of the judge's sampled outputs contained a score")
        distribution = {score: count / total for score, count in sorted(counts.items())}
        return distribution, first_scored_text

    # ------------------------------------------------------------------ #
    # MT-Bench reference: the judge model answers the question on its own
    # ------------------------------------------------------------------ #
    def _build_reference(self, question: str, conversation: str) -> _Reference:
        messages = [
            {"role": "system", "content": REFERENCE_SQL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_reference_sql_user_prompt(self._schema, conversation, question),
            },
        ]
        response = self._create(messages, max_tokens=self._max_tokens, temperature=0.0)
        sql = _extract_sql(response.choices[0].message.content or "")
        if sql.upper() == REFERENCE_NO_QUERY:
            return _Reference(
                sql=None,
                result=None,
                problem="the database does not contain the data needed to answer the question",
            )
        if not _is_single_select(sql):
            return _Reference(
                sql=sql or None, result=None, problem="no read-only SELECT query was produced"
            )
        try:
            result = self._run_read_only(sql)
        except sqlite3.Error as exc:
            logger.info("Reference query failed: %s", exc)
            return _Reference(sql=sql, result=None, problem=f"the reference query failed ({exc})")
        return _Reference(sql=sql, result=result)

    def _run_read_only(self, sql: str) -> str:
        """Run ``sql`` on a read-only connection and render the rows as text."""
        connection = sqlite3.connect(f"file:{self._sqlite_path}?mode=ro", uri=True)
        deadline = time.monotonic() + _SQL_TIMEOUT_SECONDS
        # A non-zero return value aborts the running statement.
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = connection.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(_MAX_REFERENCE_ROWS + 1)
        finally:
            connection.close()

        lines = [" | ".join(columns)]
        lines.extend(" | ".join(str(value) for value in row) for row in rows[:_MAX_REFERENCE_ROWS])
        if len(rows) > _MAX_REFERENCE_ROWS:
            lines.append(f"... (more than {_MAX_REFERENCE_ROWS} rows; remainder omitted)")
        elif not rows:
            lines.append("(no rows)")
        return _clip("\n".join(lines))

    # ------------------------------------------------------------------ #
    # OpenAI call that adapts to the parameters a model accepts
    # ------------------------------------------------------------------ #
    def _create(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        **extra: Any,
    ) -> Any:
        adaptations = 0
        while True:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                self._token_param: max_tokens,
                **extra,
            }
            if self._send_temperature:
                kwargs["temperature"] = temperature
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                adaptations += 1
                adapted = self._adapt_to_rejection(exc, uses_logprobs="logprobs" in extra)
                if not adapted or adaptations > _MAX_PARAM_ADAPTATIONS:
                    raise

    def _adapt_to_rejection(self, exc: BadRequestError, *, uses_logprobs: bool) -> bool:
        """Change the request shape after a rejection; ``False`` if nothing to change."""
        param = str(getattr(exc, "param", None) or "").lower()
        message = str(exc).lower()

        def mentions(name: str) -> bool:
            return param == name or name in message

        if mentions(_NEW_TOKEN_PARAM) or mentions(_LEGACY_TOKEN_PARAM):
            self._token_param = (
                _LEGACY_TOKEN_PARAM if self._token_param == _NEW_TOKEN_PARAM else _NEW_TOKEN_PARAM
            )
            logger.info("Model %s: switching to %r", self._model, self._token_param)
            return True
        if uses_logprobs and mentions("logprobs"):
            raise _LogprobsUnsupported from exc
        if self._send_temperature and mentions("temperature"):
            self._send_temperature = False
            logger.info("Model %s rejects temperature; using its default", self._model)
            return True
        return False


def build_judge(
    *,
    mode: str,
    api_key: str,
    model: str,
    db: Any,
    sqlite_path: Path,
    samples: int = 20,
    max_tokens: int = 600,
    low_threshold: float = 0.5,
    high_threshold: float = 0.75,
) -> AnswerJudge | None:
    """Construct the judge for ``mode``, or ``None`` when the mode is ``off``.

    ``db`` is the LangChain ``SQLDatabase`` the agent queries; its schema is
    shown to the judge.
    """
    if mode == MODE_OFF:
        logger.info("LLM-as-a-judge is off; no uncertainty estimate will be produced")
        return None
    logger.info("Building LLM-as-a-judge (mode=%s, model=%s)", mode, model)
    return AnswerJudge(
        mode=mode,
        api_key=api_key,
        model=model,
        schema=db.get_table_info(),
        sqlite_path=sqlite_path,
        samples=samples,
        max_tokens=max_tokens,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def confidence_level(confidence: float, low_threshold: float, high_threshold: float) -> str:
    """Bucket a 0..1 confidence into ``"High"``, ``"Medium"`` or ``"Low"``."""
    if confidence >= high_threshold:
        return "High"
    if confidence >= low_threshold:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Prompt inputs
# --------------------------------------------------------------------------- #
def format_sql_work(steps: Iterable[tuple[Any, Any]]) -> str:
    """Render the agent's steps for the judge: every SQL query with its result.

    Schema look-ups are reduced to one line each because the judge is shown
    the schema separately.
    """
    blocks: list[str] = []
    for index, (action, observation) in enumerate(steps, start=1):
        tool = getattr(action, "tool", "") or "unknown tool"
        tool_input = getattr(action, "tool_input", None)
        if tool == "sql_db_query":
            query = tool_input.get("query", "") if isinstance(tool_input, dict) else tool_input
            blocks.append(
                f"Step {index} - ran SQL query:\n{query}\nResult:\n{_clip(str(observation))}"
            )
        else:
            blocks.append(f"Step {index} - called {tool} with input: {_clip(str(tool_input), 200)}")
    return "\n\n".join(blocks) or "(no tool calls)"


def format_conversation(history: Sequence[Any], question: str) -> str:
    """Render the turns before ``question`` as ``User:`` / ``Assistant:`` lines."""
    messages = list(history)
    if messages and _message_text(messages[-1]) == question:
        messages.pop()  # the pending question itself
    lines = [
        f"{'User' if getattr(message, 'type', '') == 'human' else 'Assistant'}: "
        f"{_clip(_message_text(message), _MAX_HISTORY_CHARS)}"
        for message in messages[-_MAX_HISTORY_MESSAGES:]
    ]
    return "\n".join(lines) or "(none)"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _clip(text: str, limit: int = _MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... (truncated)"


# --------------------------------------------------------------------------- #
# Score extraction
# --------------------------------------------------------------------------- #
def _weighted_score(distribution: dict[int, float]) -> float:
    """G-Eval's final score: the sum of ``p(s) * s`` over all scores."""
    return sum(score * probability for score, probability in distribution.items())


def _score_in_token(token: str, low: int, high: int) -> int | None:
    """The score a single output token stands for (``"4"``, ``" 4"``, ``"[[4"``)."""
    match = _SCORE_TOKEN.fullmatch(token)
    if match is None:
        return None
    score = int(match.group(1))
    return score if low <= score <= high else None


def _distribution_from_logprobs(
    choice: Any,
    text: str,
    locate_score_token: Callable[[Sequence[Any], str], Any | None],
    scale: tuple[int, int],
) -> dict[int, float] | None:
    """Normalised probability of each score at the judge's score token."""
    tokens = getattr(getattr(choice, "logprobs", None), "content", None)
    if not tokens:
        return None
    score_token = locate_score_token(tokens, text)
    if score_token is None:
        return None

    low, high = scale
    mass: dict[int, float] = {}
    for candidate in score_token.top_logprobs or []:
        score = _score_in_token(candidate.token, low, high)
        if score is not None:
            mass[score] = mass.get(score, 0.0) + math.exp(candidate.logprob)
    total = sum(mass.values())
    if total <= 0.0:
        return None
    return {score: probability / total for score, probability in sorted(mass.items())}


def _first_score_token(tokens: Sequence[Any], low: int, high: int) -> Any | None:
    """Form-filling output: the first token that is a score."""
    for entry in tokens:
        if _score_in_token(entry.token, low, high) is not None:
            return entry
    return None


def _rating_token(tokens: Sequence[Any], text: str) -> Any | None:
    """Explanation-then-rating output: the token holding the final rating digit."""
    match = _find_rating(text)
    if match is None or "".join(entry.token for entry in tokens) != text:
        return None
    position = match.start(1)
    offset = 0
    for entry in tokens:
        end = offset + len(entry.token)
        if offset <= position < end:
            return entry
        offset = end
    return None


def _find_rating(text: str) -> re.Match[str] | None:
    """The last ``[[n]]`` in ``text`` (or, failing that, the last ``Rating: n``)."""
    for pattern in (_BRACKET_RATING, _LABELLED_RATING):
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1]
    return None


def _parse_integer_rating(text: str, low: int, high: int) -> int | None:
    match = _find_rating(text)
    if match is None or "." in match.group(1):
        return None
    rating = int(match.group(1))
    return rating if low <= rating <= high else None


def _parse_form_score(text: str, low: int, high: int) -> int | None:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    score = int(match.group())
    return score if low <= score <= high else None


# --------------------------------------------------------------------------- #
# Reference SQL
# --------------------------------------------------------------------------- #
def _extract_sql(text: str) -> str:
    """Strip Markdown fences and a trailing semicolon from the model's SQL."""
    fenced = _SQL_FENCE.search(text)
    sql = fenced.group(1) if fenced else text
    return sql.strip().rstrip(";").strip()


def _is_single_select(sql: str) -> bool:
    """Whether ``sql`` is one SELECT statement (the connection is read-only too)."""
    if not sql or ";" in sql:
        return False
    return sql.lstrip("( \n\t").upper().startswith(("SELECT", "WITH"))
