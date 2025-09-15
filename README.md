# Quiz Smith

GitHub Copilot

A lightweight Flask app that loads AI-generated MCQ (multiple choice question) JSON files, runs interactive quizzes, and includes utilities to create a Chroma vector database and generate MCQs with the Hugging Face Inference API.

This README explains how to set up the project locally, prepare required secrets, index PDF content, generate MCQs, and run the web app.

## Prerequisites

- Python 3.9+ (3.10 or newer recommended)
- Git (optional)
- A Hugging Face API key for embeddings (HF_API_KEY) and for the Inference API (HF_LLM_API_KEY) if you plan to generate MCQs via Hugging Face
- Internet access for model/API calls

## Repository layout (important files)

- app.py — Flask web app
- create_database.py — create Chroma DB from PDFs (uses Hugging Face embeddings)
- generate_mcqs.py — generate and save MCQ JSON files using Hugging Face Inference
- query_database.py — helper to query the Chroma DB
- templates/ — Jinja2 templates (index, question, results, base)
- requirements.txt — Python dependencies
- .gitignore — ignores venv, .env, chroma_db, data, __pycache__
- data/ — expected location for PDF sources (create and add PDFs)
- chroma_db/ — persistent Chroma vector DB (auto-created)

## Quick setup (local development)

1. Clone or copy the repository and cd into it.

2. Create and activate a virtual environment:
    - Linux / macOS:
      python -m venv venv
      source venv/bin/activate
    - Windows (PowerShell):
      python -m venv venv
      venv\Scripts\Activate.ps1

3. Install Python dependencies:
    pip install -r requirements.txt

4. Create a .env file in the project root with the required keys:

    Example .env (do NOT commit this file):
    HF_API_KEY=your_hf_embeddings_api_key_here
    HF_LLM_API_KEY=your_hf_inference_api_key_here
    # Optionally change Flask secret in app.py for production

    Notes:
    - create_database.py and query_database.py use `HF_API_KEY` (for embeddings).
    - generate_mcqs.py and test.py use `HF_LLM_API_KEY` (for the Inference API).
    - The code uses specific model IDs (see files). Ensure your keys have access to those models or update the model IDs.

5. Add your PDFs to `data/` (one or more .pdf files). If `data/` does not exist, create it.

## Create the Chroma vector database (index PDFs)

1. Verify your HF_API_KEY is correct.
2. Run:
    python create_database.py

What this does:
- Loads PDFs from `data/`
- Splits documents into chunks
- Uses Hugging Face endpoint embeddings to create a Chroma vector store in `./chroma_db`

Troubleshooting:
- If no PDFs are found, add PDFs to `data/`.
- If embeddings fail, ensure HF_API_KEY is set and the model (BAAI/bge-base-en-v1.5) is accessible.

## Generate MCQs (optional)

You can generate MCQs from the indexed content using the Inference API:

1. Ensure `HF_LLM_API_KEY` is present in `.env`.
2. Run the generator:
    python generate_mcqs.py

Follow the interactive prompts:
- Enter a query/topic.
- Optionally set number of questions (default 5).
- The script will query the Chroma DB, build a prompt, call the Inference API, parse and save MCQs to a JSON file named like `mcqs_<query>_<timestamp>.json`.

Notes:
- The generator expects the Chroma DB to exist and return relevant chunks.
- The output JSON files are what the web app loads to run quizzes.

## Run the web app

Start the Flask app (development):
python app.py

- The app runs by default at http://0.0.0.0:5000
- Open the URL in a browser. The home page lists available MCQ JSON files (files starting with `mcqs_*.json` in the project root).
- Click "Start Quiz" to begin.

Session data and progress:
- Quiz state is stored in Flask session (server-side secret in app.py).
- Use `/reset` to clear the session and return to home.

## Dev and debugging helpers

- test.py — quick test for the Inference client (useful to validate `HF_LLM_API_KEY`).
- query_database.py — interactive tool to search the Chroma DB and preview retrieved chunks.
- Use `python -m pip install --upgrade pip` if dependency installation errors occur.

## Environment & security

- Do not commit `.env` or API keys to version control.
- .gitignore already excludes `/venv`, `/.env`, `/chroma_db`, `/data/`.
- For production use:
  - Use a secure, random `app.secret_key`.
  - Serve behind a production WSGI server (gunicorn/uvicorn) and reverse proxy.
  - Disable Flask debug mode.

## Common troubleshooting

- "Vector database not found": run create_database.py and ensure `./chroma_db` exists and was persisted.
- Empty or poor MCQ generation: verify the DB has good content and the Inference API model is accessible; consider increasing chunks or adjusting the prompt.
- API/auth errors: check .env keys and your Hugging Face account / model access.

## Extending the project

- Add more robust parsing/validation for generated MCQ outputs.
- Add a small admin UI to upload PDFs and trigger indexing from the browser.
- Add user authentication and persistent user scores.
- Add caching for model prompts and results.

## License

This project is provided as-is (no license file included). Add a license of your choice if you plan to share publicly.

---

If you need a trimmed step-by-step for a specific OS, or a sample .env tailored to your Hugging Face access level, say which OS and provider and GitHub Copilot will provide it.