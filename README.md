# AI Exam Grader

Automated marking of handwritten student answer PDFs using **Florence-2** (vision OCR) and an **Ollama LLM** for rubric-based grading.

```
exam-grader/
├── app.py                    ← Gradio frontend (HF Spaces entry point)
├── src/
│   ├── config.py             ← Centralised config & env vars
│   ├── llm_client.py         ← Ollama streaming client
│   ├── vision.py             ← Florence-2 wrapper
│   ├── synoptic.py           ← Marking-scheme parser
│   ├── extraction.py         ← PDF → answer segments
│   ├── grading.py            ← LLM-based marks awarding
│   └── pipeline.py           ← Orchestrator
├── tests/
│   ├── test_llm_client.py
│   └── test_grading.py
├── jenkins/
│   ├── plugins.txt           ← Auto-installed Jenkins plugins
│   └── casc.yaml             ← Jenkins Configuration as Code
├── Dockerfile                ← HF Spaces container
├── docker-compose.yml        ← Local SonarQube + Jenkins stack
├── sonar-project.properties  ← SonarQube analysis config
├── Jenkinsfile               ← Declarative CI/CD pipeline
└── requirements.txt
```

---

## 1 — Run the App Locally

```bash
# Install deps
pip install -r requirements.txt

# Set your Ollama URL (or pass it in the UI)
export LLM_URL="https://your-ngrok-url.ngrok-free.app/api/generate"
export LLM_MODEL="qwen2.5:7b-instruct"

# Launch Gradio
python app.py
# → open http://localhost:7860
```

---

## 2 — Deploy to Hugging Face Spaces

### One-time setup
```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF write token
```

### Create and push the Space
```bash
# Create a Docker Space (if it doesn't exist yet)
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id="YOUR-HF-USERNAME/exam-grader",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
EOF

# Push all files
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="YOUR-HF-USERNAME/exam-grader",
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".venv", ".git", "tests"],
)
print("Deployed!")
EOF
```

Set these as **Space Secrets** in the HF UI:
| Secret | Value |
|--------|-------|
| `LLM_URL` | your Ollama/ngrok endpoint |
| `LLM_MODEL` | `qwen2.5:7b-instruct` |

---

## 3 — SonarQube + Jenkins via Docker (DevOps Lab)

### 3.1 — Start the stack

```bash
# One command brings up PostgreSQL + SonarQube + Jenkins
docker compose up -d

# Watch SonarQube boot (takes ~2 min first time)
docker compose logs -f sonarqube
```

| Service    | URL                       | Default login  |
|------------|---------------------------|----------------|
| SonarQube  | http://localhost:9000      | admin / admin  |
| Jenkins    | http://localhost:8080      | wizard on boot |

### 3.2 — First-time SonarQube setup

```bash
# 1. Open http://localhost:9000, log in (admin/admin), change password when prompted.
# 2. Create a project manually:
#      Projects → Create Project → Manually
#      Project key: exam-grader
#      Display name: AI Exam Grader
# 3. Generate a token:
#      My Account → Security → Generate Token
#      Copy the token — you'll need it next.
export SONAR_TOKEN="your-generated-token"
```

### 3.3 — Run a one-shot scan (no Jenkins needed)

```bash
# Install pytest + coverage first
pip install pytest pytest-cov

# Generate coverage report
pytest tests/ --cov=src --cov-report=xml:coverage.xml

# Run SonarScanner via the sidecar container
docker compose --profile scan run --rm sonar-scanner
```

Open **http://localhost:9000/dashboard?id=exam-grader** to see the report.

### 3.4 — Jenkins pipeline setup

```bash
# 1. Open http://localhost:8080, complete the wizard.
# 2. Install plugins (already listed in jenkins/plugins.txt):
#      Manage Jenkins → Plugins → Available → SonarQube Scanner, Blue Ocean
# 3. Add SonarQube server:
#      Manage → Configure System → SonarQube servers
#      Name: SonarQube   URL: http://sonarqube:9000
# 4. Add credentials:
#      Manage → Credentials → Global → Add
#        - Kind: Secret text, ID: SONAR_TOKEN, value: <your token>
#        - Kind: Secret text, ID: HF_TOKEN,    value: <your HF token>
# 5. Create a Pipeline job pointing to this repo's Jenkinsfile.
# 6. Build Now — watch the stages in Blue Ocean.
```

---

## 4 — Fix Issues from SonarQube Report (Assignment Step)

After the first scan, SonarQube will flag issues. Common ones and their fixes:

| Issue type | Example | Fix already applied |
|------------|---------|---------------------|
| **Code smell** | `from x import *` (wildcard import) | ✅ All imports are explicit |
| **Bug** | Bare `except:` swallowing all errors | ✅ All excepts are typed |
| **Vulnerability** | Hardcoded credentials / URLs | ✅ Moved to env vars via `config.py` |
| **Code smell** | Functions > 30 lines | ✅ Monolith split into focused modules |
| **Code smell** | Unused variables (`full_debug`) | ✅ Removed debug accumulator |

Re-run the scan after fixing to see the quality gate turn green.

---

## 5 — Marking Scheme Format

Your synoptic Excel/CSV must have these columns:

| column | example |
|--------|---------|
| `question` | `Q1` |
| `subpart` | `a` (or `-` for undivided questions) |
| `max_marks` | `5` |
| `content` | Full marking-scheme text for this part |
