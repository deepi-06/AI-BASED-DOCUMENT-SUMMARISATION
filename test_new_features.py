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
            "language": "tamil",
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

    def test_provider_requests_enforce_the_selected_language(self):
        for provider, provider_method in (
            ("Groq", "_call_groq"),
            ("Gemini", "_call_gemini"),
        ):
            response = summarizer._AIResponse(
                content="பதில்",
                provider=provider,
                model="model-a",
                input_tokens=10,
                output_tokens=5,
            )
            with (
                patch(
                    "summarizer._get_available_providers",
                    return_value=[provider],
                ),
                patch("summarizer._call_groq", return_value=response) as groq,
                patch("summarizer._call_gemini", return_value=response) as gemini,
                patch("analytics.record_ai_usage"),
            ):
                summarizer._chat(
                    "Base system prompt",
                    "Base user prompt",
                    language="tamil",
                )

            call = {"_call_groq": groq, "_call_gemini": gemini}[provider_method]
            system_prompt, user_prompt = call.call_args.args[:2]
            self.assertIn("Tamil", system_prompt)
            self.assertIn("only", system_prompt)
            self.assertIn("Tamil", user_prompt)

    def test_long_document_condensation_uses_the_selected_language(self):
        with patch("summarizer._chat", return_value="முக்கிய குறிப்புகள்") as chat:
            source, _ = summarizer.prepare_document_source(
                "Important sentence. " * 500,
                "tamil",
            )

        self.assertTrue(source)
        self.assertGreater(chat.call_count, 0)
        self.assertTrue(
            all(call.kwargs["language"] == "tamil" for call in chat.call_args_list)
        )

    def test_exam_questions_use_selected_language_and_json_structure(self):
        generated = {
            "two_mark": ["கேள்வி 1", "கேள்வி 2", "கேள்வி 3"],
            "five_mark": ["கேள்வி 4", "கேள்வி 5", "கேள்வி 6"],
            "ten_mark": ["கேள்வி 7", "கேள்வி 8", "கேள்வி 9"],
        }
        with patch(
            "summarizer._chat",
            return_value=json.dumps(generated, ensure_ascii=False),
        ) as chat:
            questions = summarizer.generate_exam_questions(
                "source document",
                language="tamil",
            )

        self.assertEqual(questions, generated)
        self.assertEqual(chat.call_args.kwargs["language"], "tamil")
        self.assertIn("selected output language", chat.call_args.args[1])

    def test_highlights_are_translated_to_the_selected_language(self):
        translated = {"highlights": ["முக்கியமான ஆய்வு முடிவுகள் தெளிவான ஆதாரத்தை காட்டுகின்றன."]}
        with patch(
            "summarizer._chat",
            return_value=json.dumps(translated, ensure_ascii=False),
        ) as chat:
            highlights = summarizer.highlight_document(
                "Important research findings show clear evidence.",
                "Important research findings",
                ["research findings"],
                {"nodes": [], "edges": []},
                language="tamil",
            )

        self.assertEqual(highlights[0]["text"], translated["highlights"][0])
        self.assertEqual(chat.call_args.kwargs["language"], "tamil")

    def test_ask_route_uses_the_document_language(self):
        with (
            patch("app._get_document_context", return_value=self.context),
            patch("app.answer_question", return_value="பதில்") as answer,
        ):
            response = self.client.post(
                "/ask",
                json={
                    "document_id": "document-1",
                    "question": "What is this about?",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["language"], "tamil")

    def test_exam_route_uses_the_document_language(self):
        with (
            patch("app._get_document_context", return_value=self.context),
            patch(
                "app.generate_exam_questions",
                return_value={
                    "two_mark": [],
                    "five_mark": [],
                    "ten_mark": [],
                },
            ) as generate_questions,
            patch("app._render_document_context", return_value="ok"),
        ):
            response = self.client.post(
                "/exam-questions",
                data={"document_id": "document-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            generate_questions.call_args.kwargs["language"],
            "tamil",
        )

    def test_dashboard_page_renders_metrics_and_charts(self):
        response = self.client.get("/page/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'data-theme="light"', response.data)
        self.assertIn(b"Total documents processed", response.data)
        self.assertIn(b"Topic performance", response.data)
        self.assertIn(b"data-nav-target=\"dashboard\"", response.data)

    def test_comparison_route_uses_and_persists_submitted_language(self):
        with self.client.session_transaction() as session:
            session["selected_language"] = "english"

        section_labels = {
            key: f"தமிழ் {key}"
            for key in (
                "overview",
                "similarities",
                "differences",
                "document_1_only",
                "document_2_only",
                "important_findings",
                "overall_summary",
                "conclusion",
                "aspect",
                "document_1",
                "document_2",
                "significance",
                "missing_information",
                "conflict",
                "different_explanation",
                "no_items",
            )
        }

        with (
            patch("app._check_ai_providers", return_value=["Groq"]),
            patch(
                "app._read_uploaded_file",
                side_effect=[("first.txt", "first document"), ("second.txt", "second document")],
            ),
            patch(
                "app.compare_documents",
                return_value={
                    "section_labels": section_labels,
                    "document_overview": {
                        "document_1": "முதல் ஆவணம்",
                        "document_2": "இரண்டாவது ஆவணம்",
                    },
                    "similarities": [],
                    "differences": [],
                    "document_1_only": [],
                    "document_2_only": [],
                    "important_findings": [],
                    "overall_summary": "ஒப்பீட்டு முடிவு",
                    "final_conclusion": "முடிவு",
                },
            ) as compare_documents,
        ):
            response = self.client.post(
                "/compare",
                data={"language": "tamil"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            compare_documents.call_args.kwargs["language"],
            "tamil",
        )
        with self.client.session_transaction() as session:
            self.assertEqual(session["selected_language"], "tamil")

    def test_comparison_generator_validates_and_localizes_report_sections(self):
        labels = {
            "overview": "கண்ணோட்டம்",
            "similarities": "ஒற்றுமைகள்",
            "differences": "வேறுபாடுகள்",
            "document_1_only": "ஆவணம் 1 மட்டும்",
            "document_2_only": "ஆவணம் 2 மட்டும்",
            "important_findings": "முக்கிய கண்டறிதல்கள்",
            "overall_summary": "ஒட்டுமொத்த சுருக்கம்",
            "conclusion": "முடிவு",
            "aspect": "அம்சம்",
            "document_1": "ஆவணம் 1",
            "document_2": "ஆவணம் 2",
            "significance": "முக்கியத்துவம்",
            "missing_information": "விடுபட்ட தகவல்",
            "conflict": "முரண்பாடு",
            "different_explanation": "மாறுபட்ட விளக்கம்",
            "no_items": "குறிப்பிடத்தக்கவை இல்லை",
        }
        response = {
            "section_labels": labels,
            "document_overview": {
                "document_1": "முதல் ஆவணம் செலவுகளை விவரிக்கிறது.",
                "document_2": "இரண்டாவது ஆவணம் அதே திட்டத்தை விளக்குகிறது.",
            },
            "similarities": ["இரண்டும் ஒரே திட்டத்தைப் பற்றி பேசுகின்றன."],
            "differences": [
                {
                    "aspect": "செலவு",
                    "document_1": "முதல் ஆவணத்தில் 10 குறிப்பிடப்பட்டுள்ளது.",
                    "document_2": "இரண்டாவது ஆவணத்தில் 12 குறிப்பிடப்பட்டுள்ளது.",
                    "significance": "இரண்டாவது மதிப்பு அதிகம்.",
                },
            ],
            "document_1_only": ["ஆரம்ப செலவு"],
            "document_2_only": ["பராமரிப்பு செலவு"],
            "important_findings": [
                {"type": "conflict", "description": "செலவு மதிப்புகள் வேறுபடுகின்றன."},
            ],
            "overall_summary": "இரு ஆவணங்களும் தொடர்புடையவை.",
            "final_conclusion": "இரண்டாவது ஆவணம் கூடுதல் விவரங்களை வழங்குகிறது.",
        }
        with patch("summarizer._chat", return_value=json.dumps(response)) as chat:
            comparison = summarizer.compare_documents(
                "Document one source",
                "Document two source",
                language="tamil",
            )

        self.assertEqual(chat.call_args.kwargs["language"], "tamil")
        self.assertEqual(comparison["section_labels"], labels)
        self.assertEqual(comparison["document_1_only"], ["ஆரம்ப செலவு"])
        self.assertEqual(
            comparison["important_findings"][0]["type_label"],
            "முரண்பாடு",
        )

    def test_quick_insights_only_return_numbers_found_in_the_source(self):
        labels = {
            key: f"தமிழ் {key}"
            for key in (
                "takeaway",
                "document_type",
                "difficulty",
                "reading_time_saved",
                "short_summary",
                "main_topics",
                "important_numbers",
                "no_numbers",
                "no_topics",
                "minute_unit",
                "reading_time_note",
                "not_available",
            )
        }
        response = {
            "labels": labels,
            "takeaway": "இந்த ஆய்வு மாணவர்களைப் பற்றியது.",
            "document_type": "ஆய்வுக் கட்டுரை",
            "short_summary": (
                "இந்த ஆய்வு மாணவர்களின் கற்றல் முறைகளைப் பரிசோதிக்கிறது. "
                "இதன் முடிவுகள் கல்வி அணுகுமுறைகளை ஒப்பிடுகின்றன."
            ),
            "difficulty": "medium",
            "difficulty_label": "நடுத்தரம்",
            "topics": ["கல்வி"],
            "important_numbers": [
                {
                    "value": "120 மாணவர்கள்",
                    "context": "ஆய்வில் பங்கேற்றவர்கள்",
                    "source_excerpt": "இந்த ஆய்வில் 120 மாணவர்கள் பங்கேற்றனர்.",
                },
                {
                    "value": "999 மாணவர்கள்",
                    "context": "ஆதாரமற்ற எண்ணிக்கை",
                    "source_excerpt": "இந்த ஆய்வில் 120 மாணவர்கள் பங்கேற்றனர்.",
                },
                {
                    "value": "120 மாணவர்கள்",
                    "context": "பங்கேற்றவர்களின் எண்ணிக்கை",
                    "source_excerpt": "மற்றொரு ஆய்வில் 120 மாணவர்கள் பங்கேற்றனர்.",
                },
            ],
        }
        with patch("summarizer._chat", return_value=json.dumps(response)) as chat:
            insights = summarizer.generate_quick_insights(
                "இந்த ஆய்வில் 120 மாணவர்கள் பங்கேற்றனர்.",
                language="tamil",
            )

        self.assertEqual(chat.call_args.kwargs["language"], "tamil")
        self.assertEqual(insights["labels"], labels)
        self.assertEqual(insights["document_type"], "ஆய்வுக் கட்டுரை")
        self.assertEqual(
            insights["short_summary"],
            "இந்த ஆய்வு மாணவர்களின் கற்றல் முறைகளைப் பரிசோதிக்கிறது. "
            "இதன் முடிவுகள் கல்வி அணுகுமுறைகளை ஒப்பிடுகின்றன.",
        )
        self.assertEqual(len(insights["important_numbers"]), 1)
        self.assertEqual(insights["important_numbers"][0]["value"], "120 மாணவர்கள்")

    def test_quick_insights_normalize_case_and_remove_repeated_topics_and_sentences(self):
        labels = {
            key: f"Insight {key.replace('_', ' ')}"
            for key in (
                "takeaway",
                "document_type",
                "difficulty",
                "reading_time_saved",
                "short_summary",
                "main_topics",
                "important_numbers",
                "no_numbers",
                "no_topics",
                "minute_unit",
                "reading_time_note",
                "not_available",
            )
        }
        response = {
            "labels": labels,
            "takeaway": "THIS REPORT EXAMINES WATER QUALITY.",
            "document_type": "RESEARCH REPORT",
            "short_summary": (
                "THIS REPORT EXAMINES WATER QUALITY. "
                "IT COMPARES TEST RESULTS FROM SEVERAL SITES. "
                "THE FINDINGS IDENTIFY AREAS FOR FURTHER STUDY."
            ),
            "difficulty": "easy",
            "difficulty_label": "EASY",
            "topics": ["Water Quality", "waterquality", "Field Testing", "Sampling"],
            "important_numbers": [],
        }
        with patch("summarizer._chat", return_value=json.dumps(response)):
            insights = summarizer.generate_quick_insights(
                "Researchers compared water quality at several sites.",
                language="english",
            )

        self.assertEqual(insights["takeaway"], "This report examines water quality.")
        self.assertEqual(insights["document_type"], "Research report")
        self.assertEqual(insights["difficulty_label"], "Easy")
        self.assertEqual(len(insights["topics"]), 3)
        self.assertEqual(insights["short_summary"].count("."), 2)
        self.assertNotIn(insights["takeaway"], insights["short_summary"])

    def test_quick_insights_endpoint_caches_localized_results_and_time_estimate(self):
        labels = {
            key: f"தமிழ் {key}"
            for key in (
                "takeaway",
                "document_type",
                "difficulty",
                "reading_time_saved",
                "short_summary",
                "main_topics",
                "important_numbers",
                "no_numbers",
                "no_topics",
                "minute_unit",
                "reading_time_note",
                "not_available",
            )
        }
        labels["minute_unit"] = "நிமிடங்கள்"
        generated = {
            "labels": labels,
            "takeaway": "ஆவணத்தின் நோக்கம்.",
            "document_type": "அறிக்கை",
            "short_summary": "இது ஆவணத்தின் சுருக்கம். இதில் முக்கிய கருத்துகள் விளக்கப்படுகின்றன.",
            "difficulty": "easy",
            "difficulty_label": "எளிது",
            "topics": ["கல்வி"],
            "important_numbers": [],
        }
        context = {
            "text": "ஒன்று இரண்டு மூன்று நான்கு ஐந்து",
            "result": {
                "summary": "சுருக்கம்",
                "analysis": {"document_type": "அறிக்கை"},
            },
            "language": "tamil",
            "quick_insights": {
                "language": "tamil",
                "takeaway": "old incomplete cached result",
            },
        }
        with (
            patch("app._get_document_context", return_value=context),
            patch("app.generate_quick_insights", return_value=generated) as generate,
        ):
            first = self.client.post(
                "/quick-insights",
                json={"document_id": "insights-test"},
            )
            second = self.client.post(
                "/quick-insights",
                json={"document_id": "insights-test"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.get_json()["document_word_count"], 5)
        self.assertEqual(first.get_json()["reading_time_saved_seconds"], 1)
        self.assertEqual(
            first.get_json()["insights"]["reading_time_saved_label"],
            "<1 நிமிடங்கள்",
        )
        self.assertEqual(first.get_json()["document_type"], "அறிக்கை")
        self.assertEqual(first.get_json()["insights"]["schema_version"], 2)
        generate.assert_called_once_with(context["text"], language="tamil")

    def test_quick_insights_sidebar_is_available(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'data-nav-target="quick-insights"', response.data)

    def test_homepage_uses_live_dashboard_statistics_and_existing_features(self):
        dashboard_data = {
            "documents_processed": 17,
            "reading_minutes_saved": 29,
            "quizzes_attempted": 4,
        }
        with patch("app.get_dashboard_data", return_value=dashboard_data) as stats:
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        stats.assert_called_once_with()
        self.assertIn(b'class="home-stat-value">17</strong>', response.data)
        self.assertIn(b'class="home-stat-value">29</strong>', response.data)
        self.assertIn(b'class="home-stat-value">4</strong>', response.data)
        self.assertIn(b'aria-label="Open Ask AI assistant"', response.data)
        self.assertIn(b'id="home-activity-list"', response.data)
        self.assertIn(b'id="upload-form"', response.data)

    def test_workspace_pages_retain_the_selected_language(self):
        context = {"language": "english"}
        with self.client.session_transaction() as session:
            session["document_id"] = "language-preference-test"

        with patch("app._get_document_context", return_value=context):
            preference = self.client.post(
                "/language",
                data={"language": "tamil"},
            )
        response = self.client.get("/page/dashboard")

        self.assertEqual(preference.status_code, 200)
        self.assertEqual(preference.get_json()["language"], "tamil")
        self.assertEqual(context["language"], "tamil")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<option value="tamil" selected>', response.data)

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
            ) as prepare_source,
            patch("app.analyze_document_bundle", return_value=result) as analyze_bundle,
            patch("app.highlight_document", return_value=[]) as highlights,
            patch("app.record_processed_document") as record_document,
        ):
            response = self.client.post(
                "/",
                data={
                    "document": (BytesIO(b"sample text"), "sample.txt"),
                    "language": "tamil",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        prepare_source.assert_called_once_with(
            "one two three four",
            "tamil",
        )
        self.assertEqual(analyze_bundle.call_args.args[2], "tamil")
        self.assertEqual(highlights.call_args.kwargs["language"], "tamil")
        recorded = record_document.call_args.args
        self.assertEqual(recorded[1], "sample.txt")
        self.assertEqual(recorded[2], 4)
        self.assertGreaterEqual(recorded[3], 0)
        with self.client.session_transaction() as session:
            self.assertEqual(session["selected_language"], "tamil")
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
            patch("app.generate_mcq_questions", return_value=self.questions) as generate_questions,
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
        self.assertEqual(
            generate_questions.call_args.kwargs["language"],
            "tamil",
        )
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
        with patch("summarizer._chat", return_value=json.dumps(generated)) as chat:
            questions = summarizer.generate_mcq_questions(
                "source text",
                count=1,
                language="tamil",
            )

        self.assertEqual(questions[0]["topic"], "Cell biology")
        self.assertEqual(questions[0]["correct_answer"], 0)
        self.assertEqual(chat.call_args.kwargs["language"], "tamil")

    def test_mcq_generator_rejects_malformed_question_data(self):
        with patch(
            "summarizer._chat",
            return_value=json.dumps({"questions": [{"question": "Missing schema"}]}),
        ):
            with self.assertRaises(summarizer.SummarizerError):
                summarizer.generate_mcq_questions("source text", count=1)


if __name__ == "__main__":
    unittest.main()
