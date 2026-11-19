# Apple FY2023 10-K Terminal AI Agent

This is a terminal application using real LLM function calling, BM25 document retrieval, and a decimal calculator. It reads the actual Apple FY2023 Form 10-K PDF rather than hardcoding the assessment answers.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and enter your own Groq API key. Never commit or share the key.

Download **Apple Inc. FY2023 Form 10-K** as a PDF and save it in this folder as `apple_2023_10k.pdf`. The assessment PDF is only the task description, **not** the report itself.

Run:

```powershell
python agent.py
```

Or supply a report PDF path:

```powershell
python agent.py "C:\path\to\apple_2023_10k.pdf"
```

Ask each of the six assessment questions, plus follow-ups like `What is the percentage decrease?`.

## Notes

- The tool searches PDF-extracted text. Some financial tables are difficult to extract accurately; inspect results and consider table-aware extraction if needed.
- Printed page detection uses standalone page numbers near page bottoms. If absent, the agent labels the PDF page **unverified** rather than inventing printed-page citations.
- The model may need several retrieval calls. Test against the six provided expected outputs.
- The agent uses a Groq OpenAI-compatible endpoint; the model must support tool calling and be available to your account.
- Do not put secrets in the code or commit `.env`.
