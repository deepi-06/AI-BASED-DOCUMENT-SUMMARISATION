import os
import re
import time
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE, override=False)

from analytics import (
    get_dashboard_data,
    init_analytics_db,
    record_processed_document,
    record_quiz_attempt,
    validate_pricing_config,
)


# ==========================================
# 2. AI API PROVIDER CONFIGURATION
# ==========================================

# Only two AI providers: Groq and Gemini.
# OpenAI has been removed.

AI_API_KEYS = {
    "Groq": os.getenv("GROQ_API_KEY", "").strip(),
    "Gemini": os.getenv("GEMINI_API_KEY", "").strip(),
}


def get_available_providers():
    """Return configured Groq and Gemini providers."""

    invalid_keys = {
        "YOUR_API_KEY_HERE",
        "your_api_key",
        "your_groq_api_key",
        "your_gemini_api_key",
    }

    return [
        provider
        for provider, key in AI_API_KEYS.items()
        if key and key not in invalid_keys
    ]


# ==========================================
# 3. IMPORT DOCUMENT AND AI FUNCTIONS
# ==========================================

from document_reader import (
    DocumentError,
    is_allowed_file,
    read_document,
)

from summarizer import (
    SUMMARY_FORMATS,
    SUMMARY_LANGUAGES,
    SUMMARY_TYPES,
    SummarizerError,
    analyze_document_bundle,
    answer_question,
    compare_documents,
    generate_exam_questions,
    generate_mcq_questions,
    highlight_document,
    prepare_document_source,
)


# ==========================================
# 4. FLASK CONFIGURATION
# ==========================================

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

MAX_FILE_MB = 10

DOCUMENT_CONTEXT_TTL_SECONDS = 30 * 60

MAX_DOCUMENT_CONTEXTS = 5

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_FILE_MB * 2 + 1
) * 1024 * 1024

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "dev-secret-change-me"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_analytics_db()
validate_pricing_config()


# ==========================================
# 5. DOCUMENT SESSION MANAGEMENT
# ==========================================

document_contexts = {}


def _prune_document_contexts():

    now = time.monotonic()

    expired = [
        token
        for token, context in document_contexts.items()
        if now - context["created_at"]
        > DOCUMENT_CONTEXT_TTL_SECONDS
    ]

    for token in expired:
        document_contexts.pop(token, None)


def _store_document_context(
    text,
    filename,
    summary_type,
    language,
    result,
    summary_format="custom",
):

    _prune_document_contexts()

    while len(document_contexts) >= MAX_DOCUMENT_CONTEXTS:

        oldest_token = next(iter(document_contexts))

        document_contexts.pop(oldest_token, None)

    token = uuid.uuid4().hex

    document_contexts[token] = {
        "created_at": time.monotonic(),
        "text": text,
        "filename": filename,
        "summary_type": summary_type,
        "language": language,
        "summary_format": summary_format,
        "result": result,
    }

    return token


def _get_document_context(token):

    _prune_document_contexts()

    if not token:
        return None

    return document_contexts.get(token)


# ==========================================
# 6. TEMPLATE RENDERING
# ==========================================

def _render_form(
    summary_type="medium",
    language="english",
    summary_format="custom",
    view="home",
    **context,
):

    active_context = {}

    if not context.get("result"):

        token = session.get("document_id")

        saved_context = _get_document_context(token)

        if saved_context is not None:

            active_context = {
                "result": saved_context["result"],
                "filename": saved_context["filename"],
                "document_id": token,
                "study_quiz": saved_context.get("study_quiz"),
                "study_quiz_result": saved_context.get("study_quiz_result"),
            }

        elif token:

            session.pop("document_id", None)

    if view == "dashboard":
        context["dashboard_data"] = get_dashboard_data()

    return render_template(
        "index.html",
        summary_types=SUMMARY_TYPES,
        selected=summary_type,
        selected_language=language,
        selected_format=summary_format,
        view=view,
        summary_formats=SUMMARY_FORMATS,
        summary_languages=SUMMARY_LANGUAGES,
        available_providers=get_available_providers(),
        **active_context,
        **context,
    )


def _render_document_context(
    token,
    context,
    **extra,
):

    return _render_form(
        summary_type=context["summary_type"],
        language=context["language"],
        summary_format=context.get(
            "summary_format",
            "custom"
        ),
        result=context["result"],
        filename=context["filename"],
        document_id=token,
        study_quiz=context.get("study_quiz"),
        study_quiz_result=context.get("study_quiz_result"),
        **extra,
    )


# ==========================================
# 7. API CONFIGURATION CHECK
# ==========================================

def _check_ai_providers():

    providers = get_available_providers()

    if not providers:

        raise SummarizerError(
            "No AI API keys are configured. "
            "Please add your GROQ_API_KEY or GEMINI_API_KEY "
            "in the .env file."
        )

    return providers


# ==========================================
# 8. HOME AND DOCUMENT UPLOAD
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return _render_form()

    summary_type = request.form.get(
        "summary_type",
        "medium"
    )

    if summary_type not in SUMMARY_TYPES:
        summary_type = "medium"

    language = request.form.get(
        "language",
        "english"
    ).lower()

    if language == "other":

        language = " ".join(
            request.form.get(
                "custom_language",
                ""
            ).split()
        )

        if not re.fullmatch(
            r"[\w][\w .+-]{1,39}",
            language,
            flags=re.UNICODE,
        ):
            language = "english"

    elif language not in SUMMARY_LANGUAGES:

        language = "english"

    summary_format = request.form.get(
        "summary_format",
        "custom"
    )

    if summary_format not in {
        *SUMMARY_FORMATS,
        "custom"
    }:
        summary_format = "custom"

    def show_error(message):

        flash(message, "error")

        return _render_form(
            summary_type,
            language,
            summary_format
        )

    # Check whether at least one API key is available.
    try:
        _check_ai_providers()

    except SummarizerError as error:
        return show_error(str(error))

    uploaded_file = request.files.get("document")

    if uploaded_file is None or uploaded_file.filename == "":

        return show_error(
            "Please choose a file to upload."
        )

    original_name = (
        secure_filename(uploaded_file.filename)
        or "document"
    )

    if not is_allowed_file(uploaded_file.filename):

        return show_error(
            "Unsupported file type. "
            "Please upload a PDF, DOCX or TXT file."
        )

    saved_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4().hex}_{original_name}",
    )
    processing_started = time.perf_counter()

    try:

        uploaded_file.save(saved_path)

        if (
            os.path.getsize(saved_path)
            > MAX_FILE_MB * 1024 * 1024
        ):

            raise DocumentError(
                f"The document must be no larger than "
                f"{MAX_FILE_MB} MB."
            )

        text = read_document(saved_path)

    except DocumentError as error:

        return show_error(str(error))

    except OSError:

        app.logger.exception(
            "Failed to save or read the uploaded document."
        )

        return show_error(
            "The uploaded file could not be read. "
            "Please try again."
        )

    finally:

        if os.path.exists(saved_path):

            try:
                os.remove(saved_path)

            except OSError:

                app.logger.exception(
                    "Failed to remove a temporary uploaded file."
                )

    # AI summarization and document analysis.
    try:

        prepared_source = prepare_document_source(text)

        result = analyze_document_bundle(
            text,
            summary_type,
            language,
            summary_format,
            prepared_source=prepared_source,
        )

    except SummarizerError as error:

        app.logger.warning(
            "Document AI request failed: %s",
            error
        )

        return show_error(str(error))

    except Exception:

        app.logger.exception(
            "Unexpected error while processing the document."
        )

        return show_error(
            "An unexpected error occurred while "
            "processing the document. Please try again."
        )

    # Generate document highlights.
    result["highlights"] = highlight_document(
        text,
        result["summary"],
        result["key_points"],
        result["graph"],
    )

    # Store document context.
    token = _store_document_context(
        text,
        original_name,
        summary_type,
        language,
        result,
        summary_format,
    )

    session["document_id"] = token
    try:
        record_processed_document(
            token,
            original_name,
            len(text.split()),
            time.perf_counter() - processing_started,
        )
    except Exception:
        app.logger.exception(
            "Could not record analytics for the processed document."
        )

    return _render_document_context(
        token,
        document_contexts[token],
        view="summary",
    )


@app.get("/page/<view>")
def workspace_page(view):
    """Render a dedicated workspace view with the current document context."""

    allowed_views = {
        "home",
        "analyzer",
        "summary",
        "listen",
        "mind-map",
        "ask-ai",
        "comparison",
        "study-mode",
        "dashboard",
        "history",
        "settings",
    }

    if view not in allowed_views:
        return "Page not found", 404

    return _render_form(view=view)


# ==========================================
# 9. ASK AI
# ==========================================

@app.route("/ask", methods=["POST"])
def ask():

    is_json_request = request.is_json

    payload = (
        request.get_json()
        if is_json_request
        else request.form
    )

    if is_json_request and not isinstance(payload, dict):

        return jsonify(
            error="The chat request must be a JSON object."
        ), 400

    token = payload.get("document_id", "")

    if not isinstance(token, str):
        token = ""

    context = _get_document_context(token)

    if context is None:

        if is_json_request:

            return jsonify(
                error="Upload and analyse a document before asking AI."
            ), 400

        flash(
            "This document session has expired. "
            "Please upload the document again.",
            "error"
        )

        return _render_form(view="ask-ai")

    question = payload.get("question", "")

    question = (
        question.strip()
        if isinstance(question, str)
        else ""
    )

    if not question:

        if is_json_request:

            return jsonify(
                error="Enter a question about the document."
            ), 400

        flash(
            "Enter a question about the document.",
            "error"
        )

        return _render_document_context(
            token,
            context,
            view="ask-ai",
            question=question
        )

    if len(question) > 1000:

        if is_json_request:

            return jsonify(
                error="Please keep your question under 1,000 characters."
            ), 400

        flash(
            "Please keep your question under 1,000 characters.",
            "error"
        )

        return _render_document_context(
            token,
            context,
            view="ask-ai",
            question=question
        )

    conversation = []

    history = (
        payload.get("history", [])
        if is_json_request
        else []
    )

    if isinstance(history, list):

        for turn in history[-8:]:

            if not isinstance(turn, dict):
                continue

            role = turn.get("role")
            content = turn.get("content")

            if (
                isinstance(role, str)
                and role in {"user", "assistant"}
                and isinstance(content, str)
            ):

                conversation.append({
                    "role": role,
                    "content": content[:2000]
                })

    try:

        answer = answer_question(
            context["text"],
            question,
            conversation
        )

    except SummarizerError as error:

        app.logger.warning(
            "Document question request failed: %s",
            error
        )

        if is_json_request:
            return jsonify(error=str(error)), 502

        flash(str(error), "error")

        return _render_document_context(
            token,
            context,
            view="ask-ai",
            question=question
        )

    except Exception:

        app.logger.exception(
            "Unexpected error while answering a document question."
        )

        if is_json_request:

            return jsonify(
                error="The question could not be answered right now. "
                "Please try again."
            ), 500

        flash(
            "The question could not be answered right now. "
            "Please try again.",
            "error"
        )

        return _render_document_context(
            token,
            context,
            view="ask-ai",
            question=question
        )

    if is_json_request:
        return jsonify(answer=answer)

    return _render_document_context(
        token,
        context,
        view="ask-ai",
        question=question,
        answer=answer
    )


# ==========================================
# 10. EXAM QUESTION GENERATION
# ==========================================

@app.route("/exam-questions", methods=["POST"])
def exam_questions():

    token = request.form.get(
        "document_id",
        ""
    )

    context = _get_document_context(token)

    if context is None:

        flash(
            "This document session has expired. "
            "Please upload the document again.",
            "error"
        )

        return _render_form(view="study-mode")

    try:

        questions = generate_exam_questions(
            context["text"]
        )

    except SummarizerError as error:

        app.logger.warning(
            "Exam question generation failed: %s",
            error
        )

        flash(str(error), "error")

        return _render_document_context(
            token,
            context,
            view="study-mode",
        )

    except Exception:

        app.logger.exception(
            "Unexpected error generating exam questions."
        )

        flash(
            "Exam questions could not be generated right now. "
            "Please try again.",
            "error"
        )

        return _render_document_context(
            token,
            context,
            view="study-mode",
        )

    return _render_document_context(
        token,
        context,
        view="study-mode",
        exam_questions=questions
    )


@app.post("/study-quiz")
def study_quiz():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="The quiz request must be a JSON object."), 400

    token = payload.get("document_id")
    if not isinstance(token, str) or not token:
        return jsonify(
            error="Upload and analyse a document before building a quiz."
        ), 400
    context = _get_document_context(token)
    if context is None:
        return jsonify(
            error="This document session has expired. Please upload it again."
        ), 410

    try:
        questions = generate_mcq_questions(context["text"])
    except SummarizerError as error:
        app.logger.warning("MCQ generation failed: %s", error)
        return jsonify(error=str(error)), 502
    except Exception:
        app.logger.exception("Unexpected error generating MCQ questions.")
        return jsonify(
            error="Quiz questions could not be generated. Please try again."
        ), 500

    quiz = {"id": uuid.uuid4().hex, "questions": questions}
    context["study_quiz"] = quiz
    context.pop("study_quiz_result", None)
    return jsonify(quiz=quiz)


@app.post("/quiz-results")
def quiz_results():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="The quiz result must be a JSON object."), 400

    token = payload.get("document_id")
    context = _get_document_context(token) if isinstance(token, str) else None
    if context is None:
        return jsonify(
            error="This document session has expired. Please upload it again."
        ), 410

    quiz = context.get("study_quiz")
    if not isinstance(quiz, dict) or payload.get("quiz_id") != quiz.get("id"):
        return jsonify(error="Generate a quiz before submitting answers."), 400

    answers = payload.get("answers")
    questions = quiz["questions"]
    if (
        not isinstance(answers, list)
        or len(answers) != len(questions)
        or any(
            answer is not None
            and (type(answer) is not int or not 0 <= answer < 4)
            for answer in answers
        )
    ):
        return jsonify(error="The submitted quiz answers are invalid."), 400

    answer_results = [
        {
            "topic": question["topic"],
            "is_correct": answer == question["correct_answer"],
        }
        for question, answer in zip(questions, answers)
    ]
    try:
        result = record_quiz_attempt(answer_results)
    except Exception:
        app.logger.exception("Could not save the quiz result.")
        return jsonify(
            error="The quiz result could not be saved. Please try again."
        ), 500

    topic_totals = {}
    for answer in answer_results:
        topic = topic_totals.setdefault(
            answer["topic"],
            {"topic": answer["topic"], "total": 0, "correct": 0},
        )
        topic["total"] += 1
        topic["correct"] += int(answer["is_correct"])
    result["topics"] = [
        {
            **topic,
            "percentage": round(topic["correct"] / topic["total"] * 100, 1),
        }
        for topic in topic_totals.values()
    ]
    result["answers"] = answers
    context["study_quiz_result"] = result
    return jsonify(result=result)


# ==========================================
# 11. DOCUMENT READING FOR COMPARISON
# ==========================================

def _read_uploaded_file(uploaded_file):

    if uploaded_file is None or not uploaded_file.filename:

        raise DocumentError(
            "Please choose both documents to compare."
        )

    if not is_allowed_file(uploaded_file.filename):

        raise DocumentError(
            "Unsupported file type. "
            "Please upload PDF, DOCX or TXT documents."
        )

    filename = (
        secure_filename(uploaded_file.filename)
        or "document"
    )

    path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4().hex}_{filename}"
    )

    try:

        uploaded_file.save(path)

        if (
            os.path.getsize(path)
            > MAX_FILE_MB * 1024 * 1024
        ):

            raise DocumentError(
                f"Each document must be no larger than "
                f"{MAX_FILE_MB} MB."
            )

        return filename, read_document(path)

    finally:

        if os.path.exists(path):

            try:
                os.remove(path)

            except OSError:

                app.logger.exception(
                    "Failed to remove a temporary uploaded file."
                )


# ==========================================
# 12. COMPARE DOCUMENTS
# ==========================================

@app.route("/compare", methods=["POST"])
def compare():

    try:

        _check_ai_providers()

    except SummarizerError as error:

        flash(str(error), "error")

        return _render_form(view="comparison")

    try:

        filename_a, text_a = _read_uploaded_file(
            request.files.get("document_a")
        )

        filename_b, text_b = _read_uploaded_file(
            request.files.get("document_b")
        )

        comparison = compare_documents(
            text_a,
            text_b
        )

    except DocumentError as error:

        flash(str(error), "error")

        return _render_form(
            view="comparison",
            comparison_error=True
        )

    except SummarizerError as error:

        app.logger.warning(
            "Document comparison failed: %s",
            error
        )

        flash(str(error), "error")

        return _render_form(
            view="comparison",
            comparison_error=True
        )

    except Exception:

        app.logger.exception(
            "Unexpected error comparing documents."
        )

        flash(
            "The documents could not be compared right now. "
            "Please try again.",
            "error"
        )

        return _render_form(
            view="comparison",
            comparison_error=True
        )

    return _render_form(
        view="comparison",
        comparison={
            **comparison,
            "filename_a": filename_a,
            "filename_b": filename_b,
        }
    )


# ==========================================
# 13. DOWNLOAD SUMMARY
# ==========================================

@app.route("/download", methods=["POST"])
def download():

    summary = request.form.get(
        "summary",
        ""
    ).strip()

    if not summary:

        flash(
            "There is no summary to download.",
            "error"
        )

        return redirect(url_for("index"))

    filename = request.form.get(
        "filename",
        "document"
    )

    summary_type = request.form.get(
        "summary_type",
        ""
    )

    language = request.form.get(
        "language",
        "English"
    )

    key_points = [
        point
        for point in request.form.getlist("key_points")
        if point.strip()
    ]

    lines = [
        "AI-BASED DOCUMENT SUMMARISATION SYSTEM",
        "=" * 40,
        f"Document     : {filename}",
        f"Summary type : {summary_type}",
        f"Language     : {language}",
        f"Generated on : {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "SUMMARY",
        "-" * 40,
        summary,
        "",
    ]

    if key_points:

        lines.extend([
            "KEY POINTS",
            "-" * 40
        ])

        lines.extend(
            f"{index}. {point}"
            for index, point in enumerate(
                key_points,
                start=1
            )
        )

        lines.append("")

    base = (
        os.path.splitext(
            secure_filename(filename)
        )[0]
        or "document"
    )

    return Response(
        "\n".join(lines),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition":
            f'attachment; filename="{base}_summary.txt"'
        },
    )


# ==========================================
# 14. FILE SIZE ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def file_too_large(_error):

    flash(
        f"The upload is too large. "
        f"Each document may be up to {MAX_FILE_MB} MB.",
        "error"
    )

    return _render_form(), 413


# ==========================================
# 15. APPLICATION STARTUP
# ==========================================

if __name__ == "__main__":

    providers = get_available_providers()

    print("\n======================================")
    print(" AI DOCUMENT SUMMARISATION SYSTEM")
    print("======================================")

    if providers:

        print("\nConfigured AI Providers:")

        for provider in providers:
            print(f" [OK] {provider} API key loaded")

    else:

        print("\n[WARNING] No AI API keys found.")

        print(
            f"Please check your .env file: {ENV_FILE}"
        )

    print("\nRunning on http://127.0.0.1:5000")

    print("======================================\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )