# AI-Based Document Summarisation System

A beginner-friendly local web app that summarises PDF, DOCX and TXT documents using AI.
Built with **Python, Flask, HTML/CSS and the Groq API**. It runs entirely on your own
computer (no cloud deployment).

## Features

- Responsive left-side workspace navigation with collapsible sidebar, theme settings, and browser-local summary history
- Upload a **PDF, DOCX or TXT** document
- Automatic text extraction (PyPDF, python-docx)
- AI summary in three lengths: **Short**, **Medium**, **Detailed**
- Executive, five-point bullet, TL;DR, or custom-length summaries
- Summary in English, Tamil, Hindi and other common languages, with a custom language option
- **Key points** list for every summary
- Document type, main topic and keywords
- Interactive knowledge graph of document concepts and relationships
- Importance-coloured extracted-text viewer linked to summary key points
- **Compare Documents** (PDF, DOCX or TXT) with similarities, additions, removals, and important differences
- **Ask AI chat** with multiple follow-up questions grounded in the uploaded document
- Study mode with key-point review, animated flashcards, timed MCQ quizzes with explanations and topic-wise scoring, self-check quizzes, and 2-mark, 5-mark and 10-mark exam question generation
- A local productivity dashboard with document, token, processing-time and quiz analytics
- **Download** the summary as a `.txt` file
- Friendly errors for empty, corrupted, password-protected, scanned (image-only) and unsupported files
- **Large documents** are split into chunks, summarised piece by piece, then combined (map-reduce)

## Project Structure

```
AI-Document-Summarisation/
├── app.py               # Flask routes (upload, summarise, download)
├── analytics.py         # Local SQLite usage and quiz analytics
├── summarizer.py        # Groq API calls, chunking, prompts
├── document_reader.py   # Text extraction for PDF / DOCX / TXT
├── requirements.txt     # Python dependencies
├── .env.example         # Template for your API key
├── README.md
├── templates/
│   └── index.html       # Web page
├── static/
│   └── style.css        # Styling
└── uploads/             # Temporary storage (files are deleted after reading)
```

## Setup in VS Code

### 1. Requirements
- Python 3.9 or newer (https://www.python.org/downloads/ - tick "Add Python to PATH" on Windows)
- VS Code with the **Python** extension
- A free Groq API key: https://console.groq.com/keys

### 2. Open the project
Unzip the folder, then in VS Code: **File > Open Folder...** and choose `AI-Document-Summarisation`.

### 3. Create a virtual environment
Open the terminal in VS Code (**Terminal > New Terminal**) and run:

**Windows (PowerShell)**
```
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process Bypass` first,
or use Command Prompt and run `venv\Scripts\activate.bat`.

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
```

When VS Code asks about the new environment, click **Yes**. Otherwise press
`Ctrl+Shift+P` > **Python: Select Interpreter** > choose the one inside `venv`.

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Add your API key
Copy `.env.example` to a new file named `.env`:

**Windows:** `copy .env.example .env`  
**macOS / Linux:** `cp .env.example .env`

Open `.env` and replace the placeholder with your real key:
```
GROQ_API_KEY=gsk_your_real_key_here
```
Never share or commit the `.env` file (it is already listed in `.gitignore`).

### 6. Run the app
```
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

## How to Use

1. Upload a PDF, DOCX or TXT document (max 10 MB).
2. Choose Executive Summary, Bullet Points, TL;DR, or a custom Short/Medium/Detailed summary, then select the output language.
3. Use the sidebar to move between the summary, analysis, knowledge map, Ask AI, study tools, and document comparison.
4. Review the document insights, explore the clickable knowledge graph, and follow the green/yellow/neutral sentence highlights to related key points.
5. Ask document-grounded questions or open Study mode to review key points, practise flashcards, and generate exam questions for a self-check quiz.
6. Use **Compare Documents** to compare another pair of PDF, DOCX or TXT files (max 10 MB each).
7. Open **Dashboard** for locally recorded usage metrics and quiz performance, or **History** for summaries saved in this browser. Use **Settings** to switch the light/dark appearance.
8. Download the generated summary as a `.txt` file if desired.

## Changing the AI Model

The default model is `openai/gpt-oss-120b`. Groq updates its model list regularly
(older models such as `llama-3.3-70b-versatile` are being retired), so the model is
configurable. To switch, add this line to `.env`:

```
GROQ_MODEL=openai/gpt-oss-20b
```
`openai/gpt-oss-20b` is a faster, lighter option. See the current list at
https://console.groq.com/docs/models.

## How Large Documents Are Handled

1. Text is split into chunks of about 8,000 characters, cut at line or sentence boundaries.
2. Each chunk is condensed into short notes.
3. The notes are combined (and condensed again if still too long).
4. The final Short / Medium / Detailed summary and key points are written from those notes.

The number of chunks is displayed with the result. The summary, document analysis,
and knowledge map are returned from one final Groq request, reusing the prepared
document text, to avoid waiting on separate requests for each result.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Groq API key is missing" | Make sure the file is named `.env` (not `.env.example`) and contains your real key. Restart the app after editing. |
| "Groq rejected the API key" | The key is wrong or revoked. Create a new one in the Groq console. |
| "model ... may no longer be available" | Set a currently supported model in `GROQ_MODEL`. |
| "Rate limit reached" | Free tiers have limits. Wait a minute or try a smaller document. |
| "No text could be extracted from this PDF" | The PDF is a scan (images only). OCR is not supported; use a text-based PDF. |
| `ModuleNotFoundError` | The virtual environment is not active or dependencies are not installed. Repeat steps 3 and 4. |
| Old `.doc` files | Save them as `.docx` in Word first. |

## Notes

- Uploaded files are deleted from `uploads/` right after the text is extracted.
- Extracted text is kept in the running app's memory for up to 30 minutes for follow-up questions and exam question generation. The app holds at most five document sessions; restarting the app clears them.
- Dashboard analytics are stored locally in `instance/dashboard.sqlite3`; document contents are not stored in the analytics database. Reading time saved is estimated from extracted word count at 200 words per minute, less the measured processing duration. Token counts come from provider-reported usage metadata.
- Analytics begin recording from this version onward; previous browser history does not contain the backend token or timing data needed to backfill older dashboard totals.
- API cost is shown only when exact per-model rates are configured. Set `AI_MODEL_PRICING_JSON` in `.env` using current provider rates, for example `{"your-model-id":{"input_usd_per_million":0.0,"output_usd_per_million":0.0}}`, replacing the model ID and sample numeric values with published rates. Without configured rates, the dashboard displays a dash rather than a fabricated estimate.
- Document comparisons use Groq to identify material content changes from both extracted documents. Long inputs are condensed to fit API context limits, so the comparison is an AI analysis rather than a line-by-line legal or archival diff.
- The knowledge map is rendered as a local interactive SVG; no graph package or external visualization service is required.
- Document text is sent to Groq's API for summarisation, so do not upload confidential
  documents unless that is acceptable under your own policies.
- `app.py` runs Flask in debug mode for local development only; do not use it for public hosting.
