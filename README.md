# AI Question Generator & Duplicate Checker

A Streamlit application that generates questions, detects duplicates, and stores accepted questions in SQLite.

Built for **Question 2: AI Question Generation and Duplicate Prevention**.

## Features

- **Generation-time avoidance:** Uses relevant, diverse existing questions to guide the AI toward different assessment objectives.
- **Independent verification:** Checks every generated draft before accepting it.
- **Exact duplicate detection:** Normalizes case, spacing, and common punctuation.
- **Semantic duplicate detection:** Retrieves similar questions using local embeddings, then compares their meaning with an AI verifier.
- **Persistent storage:** Saves questions, embeddings, tags, and classifications in SQLite.
- **Review queue:** Keeps uncertain submissions separate from accepted questions.
- **Evaluation and logs:** Includes reflection test cases, an activity log, and optional Logfire tracing.

## Architecture

**Generation:**  
Subject → Retrieve existing examples → Generate draft → Independently check → Save, review, or retry

**Detection:**  
Candidate → Normalized exact lookup → Embedding retrieval → AI verification

| Component | Technology |
|---|---|
| Interface | Streamlit |
| Generation and verification | Groq — `openai/gpt-oss-20b` |
| Local embeddings | Sentence Transformers — `all-MiniLM-L6-v2` |
| Similarity calculations | NumPy |
| Storage | SQLite |
| Response validation | Pydantic |
| Observability | Logfire and session activity logs |

Tags are flexible key–value metadata. Duplicate decisions are based on **question content**, not tag equality.

## Getting Started

Requires **Python 3.10+** and a **Groq API key**.

### 1. Install dependencies

```bash terminal.sh
python -m pip install streamlit groq sentence-transformers numpy pydantic python-dotenv logfire tiktoken
```

### 2. Configure environment variables

Create `.env` beside `main.py`:

```dotenv .env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
LOGFIRE_TOKEN=
```

`LOGFIRE_TOKEN` is optional. Never commit your actual `.env` file.

### 3. Run

```bash terminal.sh
python -m streamlit run main.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal, usually **http://localhost:8501**.

The first run downloads the embedding model and creates `questions.db`.

## Try It

1. Click **Import reflection examples**.
2. Open **Question library** and classify older/imported questions.
3. Check an exact duplicate:  
   **“DEFINE the reflection of light!!!”**
4. Check a semantic paraphrase:  
   **“Describe what it means when light is reflected.”**
5. Generate a question and inspect its verification result.
6. Run the **Evaluation** tab to compare predictions with the supplied Q1–Q10 labels.

Newly accepted questions receive saved classifications automatically. The fixture import intentionally includes historical duplicates; normal submissions are screened before saving.

## Duplicate Policy

Questions are duplicates when they test substantially the same knowledge or reasoning under equivalent conditions.

- Paraphrases count as duplicates.
- Shared subject alone does not imply duplication.
- Changing only numbers in the same calculation template counts as duplication.
- Ambiguous questions may require human review.
- Generation stops after three unsuccessful attempts.

## Evaluation and Limitations

The included evaluation checks each fixture against earlier fixtures in a temporary database. It reports classification-label agreement and timing without modifying the saved collection.

This is a **working prototype**, not a guarantee of perfect detection:

- Semantic verification examines retrieved neighbors and can miss other matches.
- Prompt token budgeting is approximate.
- Generation and verification use the same model and may share errors.
- SQLite serializes final checking and saving; concurrent writers may wait during AI calls.
- The ten-question demonstration is not a comprehensive accuracy benchmark.

Future evaluation should measure duplicate precision, recall, retrieval recall, review rate, and latency on a larger human-labeled dataset.

## Privacy

Embeddings run locally, but candidate questions and selected existing questions are sent to Groq. Optional Logfire integration sends configured telemetry.

Do not upload confidential question content without authorization, and keep API keys out of source control.
~~~

You can use the **Apply Button**, or switch to **Agent Mode** using the Mode Selector dropdown to create the file automatically.
