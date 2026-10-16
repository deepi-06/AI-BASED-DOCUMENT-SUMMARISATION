"""Persistent, local analytics for document processing and study sessions."""

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.abspath(
    os.getenv(
        "ANALYTICS_DB_PATH",
        os.path.join(BASE_DIR, "instance", "dashboard.sqlite3"),
    )
)
def _connect(database_path=None):
    path = database_path or DATABASE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pricing_rates():
    raw = os.getenv("AI_MODEL_PRICING_JSON", "").strip()
    if not raw:
        return {}

    try:
        pricing = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("AI_MODEL_PRICING_JSON must contain valid JSON.") from error

    if not isinstance(pricing, dict):
        raise ValueError("AI_MODEL_PRICING_JSON must be a JSON object keyed by model.")

    validated = {}
    for model, rates in pricing.items():
        if not isinstance(model, str) or not model.strip() or not isinstance(rates, dict):
            raise ValueError("Each model pricing entry must be an object.")
        model_rates = {}
        for key in ("input_usd_per_million", "output_usd_per_million"):
            value = rates.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Model pricing for {model!r} requires a non-negative "
                    f"numeric {key} value."
                )
            model_rates[key] = float(value)
        validated[model] = model_rates
    return validated


def validate_pricing_config():
    """Raise a clear startup error if optional model rates are malformed."""
    _pricing_rates()


def init_analytics_db(database_path=None):
    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                word_count INTEGER NOT NULL CHECK (word_count >= 0),
                processing_seconds REAL NOT NULL CHECK (processing_seconds >= 0)
            );

            CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                estimated_cost_usd REAL
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id TEXT PRIMARY KEY,
                occurred_at TEXT NOT NULL,
                score INTEGER NOT NULL CHECK (score >= 0),
                total INTEGER NOT NULL CHECK (total > 0),
                score_percent REAL NOT NULL CHECK (score_percent >= 0 AND score_percent <= 100)
            );

            CREATE TABLE IF NOT EXISTS quiz_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                topic TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1))
            );
            """
        )


def record_processed_document(
    document_id,
    filename,
    word_count,
    processing_seconds,
    database_path=None,
):
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO processed_documents
                (id, filename, processed_at, word_count, processing_seconds)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                _utc_now(),
                max(0, int(word_count)),
                max(0.0, float(processing_seconds)),
            ),
        )


def record_ai_usage(
    provider,
    model,
    input_tokens,
    output_tokens,
    database_path=None,
):
    """Persist provider-reported token usage and an optional configured estimate."""
    rates = _pricing_rates().get(model)
    estimated_cost = None
    if (
        rates is not None
        and input_tokens is not None
        and output_tokens is not None
    ):
        estimated_cost = (
            input_tokens * rates["input_usd_per_million"]
            + output_tokens * rates["output_usd_per_million"]
        ) / 1_000_000

    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ai_usage
                (occurred_at, provider, model, input_tokens, output_tokens, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                provider,
                model,
                input_tokens,
                output_tokens,
                estimated_cost,
            ),
        )


def record_quiz_attempt(answers, database_path=None):
    attempt_id = uuid.uuid4().hex
    correct_count = sum(bool(answer["is_correct"]) for answer in answers)
    total = len(answers)
    score_percent = correct_count / total * 100

    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO quiz_attempts (id, occurred_at, score, total, score_percent)
            VALUES (?, ?, ?, ?, ?)
            """,
            (attempt_id, _utc_now(), correct_count, total, score_percent),
        )
        connection.executemany(
            """
            INSERT INTO quiz_answers (attempt_id, topic, is_correct)
            VALUES (?, ?, ?)
            """,
            [
                (attempt_id, answer["topic"], int(bool(answer["is_correct"])))
                for answer in answers
            ],
        )

    return {
        "attempt_id": attempt_id,
        "correct": correct_count,
        "wrong": total - correct_count,
        "total": total,
        "percentage": round(score_percent, 1),
    }


def get_dashboard_data(database_path=None):
    init_analytics_db(database_path)
    with _connect(database_path) as connection:
        documents = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(word_count), 0) AS words,
                   COALESCE(SUM(processing_seconds), 0) AS processing_seconds,
                   AVG(processing_seconds) AS average_processing_seconds
            FROM processed_documents
            """
        ).fetchone()
        quizzes = connection.execute(
            """
            SELECT COUNT(*) AS attempts, AVG(score_percent) AS average_score
            FROM quiz_attempts
            """
        ).fetchone()
        usage = connection.execute(
            """
            SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0)
                       AS total_tokens,
                   COUNT(*) AS calls
            FROM ai_usage
            """
        ).fetchone()
        usage_details = connection.execute(
            """
            SELECT model, input_tokens, output_tokens, estimated_cost_usd
            FROM ai_usage
            """
        ).fetchall()
        topics = connection.execute(
            """
            SELECT topic, COUNT(*) AS total,
                   SUM(is_correct) AS correct,
                   ROUND(AVG(is_correct) * 100, 1) AS percentage
            FROM quiz_answers
            GROUP BY topic
            ORDER BY total DESC, topic COLLATE NOCASE
            LIMIT 8
            """
        ).fetchall()
        daily_documents = connection.execute(
            """
            SELECT date(processed_at) AS day, COUNT(*) AS total
            FROM processed_documents
            WHERE date(processed_at) >= date('now', '-6 days')
            GROUP BY date(processed_at)
            """
        ).fetchall()
        daily_quizzes = connection.execute(
            """
            SELECT date(occurred_at) AS day, COUNT(*) AS total
            FROM quiz_attempts
            WHERE date(occurred_at) >= date('now', '-6 days')
            GROUP BY date(occurred_at)
            """
        ).fetchall()

    reading_minutes = documents["words"] / 200
    processing_minutes = documents["processing_seconds"] / 60
    reading_minutes_saved = max(0.0, reading_minutes - processing_minutes)
    efficiency = (
        reading_minutes_saved / reading_minutes * 100
        if reading_minutes
        else 0.0
    )
    average_quiz_score = quizzes["average_score"]
    productivity_inputs = []
    if documents["total"]:
        productivity_inputs.append(efficiency)
    if quizzes["attempts"]:
        productivity_inputs.append(average_quiz_score)
    productivity = (
        sum(productivity_inputs) / len(productivity_inputs)
        if productivity_inputs
        else 0.0
    )

    document_by_day = {row["day"]: row["total"] for row in daily_documents}
    quiz_by_day = {row["day"]: row["total"] for row in daily_quizzes}
    today = datetime.now(timezone.utc).date()
    daily_activity = []
    for days_ago in range(6, -1, -1):
        day = (today - timedelta(days=days_ago)).isoformat()
        daily_activity.append(
            {
                "date": day,
                "label": datetime.fromisoformat(day).strftime("%a"),
                "documents": document_by_day.get(day, 0),
                "quizzes": quiz_by_day.get(day, 0),
            }
        )

    calls = usage["calls"]
    pricing = _pricing_rates()
    estimated_cost = 0.0
    cost_available = calls > 0
    for call in usage_details:
        rates = pricing.get(call["model"])
        if (
            rates is not None
            and call["input_tokens"] is not None
            and call["output_tokens"] is not None
        ):
            estimated_cost += (
                call["input_tokens"] * rates["input_usd_per_million"]
                + call["output_tokens"] * rates["output_usd_per_million"]
            ) / 1_000_000
        elif call["estimated_cost_usd"] is not None:
            estimated_cost += call["estimated_cost_usd"]
        else:
            cost_available = False
            break
    return {
        "documents_processed": documents["total"],
        "reading_minutes_saved": round(reading_minutes_saved, 1),
        "efficiency_percent": round(efficiency, 1),
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": round(estimated_cost, 6)
        if cost_available
        else None,
        "average_processing_seconds": round(documents["average_processing_seconds"], 1)
        if documents["average_processing_seconds"] is not None
        else None,
        "quizzes_attempted": quizzes["attempts"],
        "average_quiz_score": round(average_quiz_score, 1)
        if average_quiz_score is not None
        else None,
        "productivity_percent": round(productivity, 1),
        "daily_activity": daily_activity,
        "topic_results": [dict(row) for row in topics],
    }
