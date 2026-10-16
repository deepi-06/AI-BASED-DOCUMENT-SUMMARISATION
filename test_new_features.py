import json
import os
import tempfile
import time
import unittest
from io import BytesIO
from unittest.mock import patch

import analytics
import app as flask_app
import summarizer


class AnalyticsTests(unittest.TestCase):
    def test_dashboard_does_not_show_a_zero_cost_before_usage_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            dashboard = analytics.get_dashboard_data(
                os.path.join(directory, "dashboard.sqlite3")
            )

        self.assertIsNone(dashboard["estimated_cost_usd"])
        self.assertEqual(dashboard["total_tokens"], 0)

    def test_dashboard_aggregates_real_document_usage_and_quiz_data(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "dashboard.sqlite3")
            analytics.init_analytics_db(database_path)
            with patch.dict(
                os.environ,
                {
                    "AI_MODEL_PRICING_JSON": json.dumps(
                        {
                            "model-a": {
                                "input_usd_per_million": 2,
                                "output_usd_per_million": 4,
                            }
                        }
                    )
                },
            ):
                analytics.record_ai_usage(
                    "Groq", "model-a", 100, 50, database_path
                )
            analytics.record_processed_document(
                "doc-1", "notes.txt", 400, 60, database_path
            )
            analytics.record_quiz_attempt(
                [
                    {"topic": "Cells", "is_correct": True},
                    {"topic": "Cells", "is_correct": False},
                ],
                database_path,
            )

            dashboard = analytics.get_dashboard_data(database_path)

        self.assertEqual(dashboard["documents_processed"], 1)
        self.assertEqual(dashboard["total_tokens"], 150)
        self.assertEqual(dashboard["estimated_cost_usd"], 0.0004)
        self.assertEqual(dashboard["reading_minutes_saved"], 1)
        self.assertEqual(dashboard["efficiency_percent"], 50)
        self.assertEqual(dashboard["average_processing_seconds"], 60)
        self.assertEqual(dashboard["quizzes_attempted"], 1)
        self.assertEqual(dashboard["average_quiz_score"], 50)
        self.assertEqual(dashboard["topic_results"][0]["topic"], "Cells")
        self.assertEqual(dashboard["topic_results"][0]["percentage"], 50)

    def test_cost_is_unavailable_without_model_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "dashboard.sqlite3")
            analytics.init_analytics_db(database_path)
            with patch.dict(os.environ, {"AI_MODEL_PRICING_JSON": ""}):
                analytics.record_ai_usage(
                    "Gemini", "model-b", 15, 5, database_path
                )
            dashboard = analytics.get_dashboard_data(database_path)

        self.assertIsNone(dashboard["estimated_cost_usd"])
        self.assertEqual(dashboard["total_tokens"], 20)

    def test_cost_can_be_estimated_after_rates_are_added(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "dashboard.sqlite3")
            analytics.init_analytics_db(database_path)
            with patch.dict(os.environ, {"AI_MODEL_PRICING_JSON": ""}):
                analytics.record_ai_usage(
                    "Groq", "model-a", 100, 50, database_path
                )
            with patch.dict(
                os.environ,
                {
                    "AI_MODEL_PRICING_JSON": json.dumps(
                        {
                            "model-a": {
                                "input_usd_per_million": 2,
                                "output_usd_per_million": 4,
                            }
                        }
                    )
                },
            ):
                dashboard = analytics.get_dashboard_data(database_path)

        self.assertEqual(dashboard["estimated_cost_usd"], 0.0004)


class StudyFeatureTests(unittest.TestCase):
    def setUp(self):
        flask_app.app.config.update(TESTING=True)
        self.client = flask_app.app.test_client()
        self.context = {
            "text": "Cell biology document text.",
            "study_quiz": None,
        }
        self.questions = [
            {
                "question": "What is the primary unit of life?",
                "options": ["Cell", "Atom", "Organ", "Tissue"],
                "correct_answer": 0,
                "explanation": "The cell is the basic structural and functional unit of life.",
                "topic": "Cell biology",
            },
            {
                "question": "Which structure holds genetic material?",
                "options": ["Nucleus", "Ribosome", "Membrane", "Vacuole"],
                "correct_answer": 0,
                "explanation": "In eukaryotic cells, the nucleus contains most genetic material.",
                "topic": "Cell biology",
            },
        ]

    def test_ai_provider_usage_is_recorded(self):
        response = summarizer._AIResponse(
            content="summary",
            provider="Groq",
            model="model-a",
            input_tokens=120,
            output_tokens=30,
        )
        with (
            patch("summarizer._get_available_providers", return_value=["Groq"]),
            patch("summarizer._call_groq", return_value=response),
            patch("analytics.record_ai_usage") as record_usage,
        ):
            output = summarizer._chat("system", "prompt")

        self.assertEqual(output, "summary")
        record_usage.assert_called_once_with("Groq", "model-a", 120, 30)

    def test_dashboard_page_renders_metrics_and_charts(self):
        response = self.client.get("/page/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Total documents processed", response.data)
        self.assertIn(b"Topic performance", response.data)
        self.assertIn(b"data-nav-target=\"dashboard\"", response.data)

    def test_document_processing_records_actual_word_count_and_duration(self):
        result = {
            "summary": "A sample summary.",
            "summary_type": "medium",
            "language": "English",
            "word_count": 4,
            "key_points": ["A sample key point."],
            "analysis": {
                "document_type": "Notes",
                "main_topic": "Sample topic",
                "keywords": ["sample"],
            },
            "graph": {"nodes": [], "edges": []},
        }
        with (
            patch("app._check_ai_providers", return_value=["Groq"]),
            patch("app.read_document", return_value="one two three four"),
            patch(
                "app.prepare_document_source",
                return_value=("one two three four", 1),
            ),
            patch("app.analyze_document_bundle", return_value=result),
            patch("app.highlight_document", return_value=[]),
            patch("app.record_processed_document") as record_document,
        ):
            response = self.client.post(
                "/",
                data={
                    "document": (BytesIO(b"sample text"), "sample.txt"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        recorded = record_document.call_args.args
        self.assertEqual(recorded[1], "sample.txt")
        self.assertEqual(recorded[2], 4)
        self.assertGreaterEqual(recorded[3], 0)
        with self.client.session_transaction() as session:
            flask_app.document_contexts.pop(session.get("document_id"), None)

    def test_study_page_renders_mcq_controls_and_flashcards(self):
        token = "study-template-test"
        flask_app.document_contexts[token] = {
            **self.context,
            "created_at": time.monotonic(),
            "filename": "biology.txt",
            "summary_type": "medium",
            "language": "english",
            "summary_format": "custom",
            "result": {
                "summary": "A short document summary.",
                "summary_type": "medium",
                "language": "English",
                "word_count": 5,
                "key_points": ["Cells are the basic unit of life."],
                "analysis": {
                    "document_type": "Notes",
                    "main_topic": "Cell biology",
                    "keywords": ["cells"],
                },
                "graph": {"nodes": [], "edges": []},
                "highlights": [],
            },
            "study_quiz": {"id": "quiz-1", "questions": self.questions},
        }
        try:
            with self.client.session_transaction() as session:
                session["document_id"] = token
            response = self.client.get("/page/study-mode")
        finally:
            flask_app.document_contexts.pop(token, None)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"mcq-previous", response.data)
        self.assertIn(b"mcq-next", response.data)
        self.assertIn(b"mcq-submit", response.data)
        self.assertIn(b"mcq-timer", response.data)
        self.assertIn(b"flashcard-inner", response.data)

    def test_mcq_generation_and_quiz_submission_are_backend_connected(self):
        with (
            patch("app._get_document_context", return_value=self.context),
            patch("app.generate_mcq_questions", return_value=self.questions),
            patch(
                "app.record_quiz_attempt",
                return_value={
                    "attempt_id": "attempt-1",
                    "correct": 1,
                    "wrong": 1,
                    "total": 2,
                    "percentage": 50,
                },
            ),
        ):
            generated = self.client.post(
                "/study-quiz",
                json={"document_id": "document-1"},
            )
            quiz = generated.get_json()["quiz"]
            submitted = self.client.post(
                "/quiz-results",
                json={
                    "document_id": "document-1",
                    "quiz_id": quiz["id"],
                    "answers": [0, 1],
                },
            )

        self.assertEqual(generated.status_code, 200)
        self.assertEqual(len(quiz["questions"]), 2)
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(submitted.get_json()["result"]["correct"], 1)
        self.assertEqual(
            submitted.get_json()["result"]["topics"][0]["percentage"],
            50,
        )
        self.assertEqual(self.context["study_quiz_result"]["answers"], [0, 1])

    def test_quiz_rejects_invalid_answers(self):
        self.context["study_quiz"] = {"id": "quiz-1", "questions": self.questions}
        with patch("app._get_document_context", return_value=self.context):
            response = self.client.post(
                "/quiz-results",
                json={
                    "document_id": "document-1",
                    "quiz_id": "quiz-1",
                    "answers": [4, None],
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid", response.get_json()["error"])

    def test_mcq_generator_validates_questions(self):
        generated = {
            "questions": [
                {
                    **self.questions[0],
                }
            ]
        }
        with patch("summarizer._chat", return_value=json.dumps(generated)):
            questions = summarizer.generate_mcq_questions("source text", count=1)

        self.assertEqual(questions[0]["topic"], "Cell biology")
        self.assertEqual(questions[0]["correct_answer"], 0)

    def test_mcq_generator_rejects_malformed_question_data(self):
        with patch(
            "summarizer._chat",
            return_value=json.dumps({"questions": [{"question": "Missing schema"}]}),
        ):
            with self.assertRaises(summarizer.SummarizerError):
                summarizer.generate_mcq_questions("source text", count=1)


if __name__ == "__main__":
    unittest.main()
