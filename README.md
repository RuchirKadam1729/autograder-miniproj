# AI Exam Grader

Automated marking of handwritten student answer PDFs using **Florence-2** (vision OCR) + **Groq LLaMA 3.3** (rubric grading).

```
exam-grader/
├── app.py                    ← Gradio frontend (HF Spaces entry point)
├── src/
│   ├── config.py             ← Centralised config & env vars
│   ├── llm_client.py         ← Groq API client
│   ├── vision.py             ← Florence-2 wrapper
│   ├── synoptic.py           ← Marking-scheme parser
│   ├── extraction.py         ← PDF → answer segments
│   ├── grading.py            ← LLM-based marks awarding
│   └── pipeline.py           ← Orchestrator
├── tests/
├── jenkins/
├── Dockerfile
├── docker-compose.yml        ← SonarQube + Jenkins stack
├── sonar-project.properties
├── Jenkinsfile
└── requirements.txt
```

---

## 1 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com) and sign up (free)
2. Create an API key → copy it (`gsk_...`)
3. Set it: `export GROQ_API_KEY="gsk_..."`

---

## 2 — Run Locally

```bash
pip install -r requirements.txt

export GROQ_API_KEY="gsk_your_key_here"
python app.py
# → open http://localhost:7860
```

---

## 3 — Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login
```

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
# Create space (once)
api.create_repo(
    repo_id="YOUR-USERNAME/exam-grader",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
# Push code
api.upload_folder(
    folder_path=".",
    repo_id="YOUR-USERNAME/exam-grader",
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".venv", ".git", "tests"],
)
print("Deployed!")
EOF
```

Then in the HF Space settings → **Secrets**, add:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | your `gsk_...` key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |

---

## 4 — SonarQube + Jenkins (DevOps Lab)

### Start the stack

```bash
docker compose up -d
# SonarQube → http://localhost:9000  (admin/admin)
# Jenkins   → http://localhost:8080
```

### Run a scan (no Jenkins needed)

```bash
# Generate coverage
pytest tests/ --cov=src --cov-report=xml:coverage.xml

# Set your SonarQube token
export SONAR_TOKEN="sqp_your_token_here"

# Scan
docker compose --profile scan run --rm sonar-scanner
```

Open **http://localhost:9000/dashboard?id=exam-grader**

### Jenkins pipeline

Add these credentials in Jenkins → Manage → Credentials:

| ID | Type | Value |
|----|------|-------|
| `SONAR_TOKEN` | Secret text | SonarQube token |
| `GROQ_API_KEY` | Secret text | your Groq key |
| `HF_TOKEN` | Secret text | HF write token |

Create a Pipeline job → point to this repo → Build Now.

---

## 5 — Groq Model Options

| Model | Speed | Context | Best for |
|-------|-------|---------|----------|
| `llama-3.3-70b-versatile` | Fast | 128k | Best quality (default) |
| `llama-3.1-8b-instant` | Very fast | 128k | High volume, lower cost |
| `mixtral-8x7b-32768` | Fast | 32k | Long marking schemes |

---

## 6 — Marking Scheme Format

Your synoptic `.xlsx` / `.csv` must have:

| column | example |
|--------|---------|
| `question` | `Q1` |
| `subpart` | `a` (or `-` for whole questions) |
| `max_marks` | `5` |
| `content` | Full marking scheme text for this part |