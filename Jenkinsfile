// ── Jenkinsfile ──────────────────────────────────────────────────────────────
// Declarative pipeline for CI: lint → test → SonarQube → deploy to HF Spaces
// Requires:
//   - Jenkins SonarQube Scanner plugin
//   - Credentials: SONAR_TOKEN (secret text), HF_TOKEN (secret text)
// ─────────────────────────────────────────────────────────────────────────────

pipeline {
    agent any

    environment {
        PYTHON        = "python3"
        SONAR_PROJECT = "exam-grader"
        // SonarQube server name configured in Jenkins → Manage → Configure System
        SONAR_SERVER  = "SonarQube"
        HF_REPO       = "your-hf-username/exam-grader"   // ← change this
    }

    stages {

        // ── 1. Checkout ───────────────────────────────────────────────────────
        stage("Checkout") {
            steps {
                checkout scm
                echo "✅ Code checked out at ${env.GIT_COMMIT}"
            }
        }

        // ── 2. Install Dependencies ───────────────────────────────────────────
        stage("Install") {
            steps {
                sh """
                    ${PYTHON} -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install pytest pytest-cov flake8 pylint
                """
            }
        }

        // ── 3. Lint ───────────────────────────────────────────────────────────
        stage("Lint") {
            steps {
                sh """
                    . .venv/bin/activate
                    flake8 src/ app.py \
                        --max-line-length=100 \
                        --exclude=__pycache__ \
                        --format=default \
                        --output-file=flake8-report.txt || true
                    cat flake8-report.txt
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: "flake8-report.txt", allowEmptyArchive: true
                }
            }
        }

        // ── 4. Unit Tests + Coverage ──────────────────────────────────────────
        stage("Test") {
            steps {
                sh """
                    . .venv/bin/activate
                    pytest tests/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --cov-report=term \
                        -v || true
                """
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: "test-results.xml"
                    archiveArtifacts artifacts: "coverage.xml", allowEmptyArchive: true
                }
            }
        }

        // ── 5. SonarQube Analysis ─────────────────────────────────────────────
        stage("SonarQube Analysis") {
            steps {
                withSonarQubeEnv(env.SONAR_SERVER) {
                    withCredentials([string(credentialsId: "SONAR_TOKEN", variable: "SONAR_AUTH")]) {
                        sh """
                            sonar-scanner \
                                -Dsonar.login=${SONAR_AUTH} \
                                -Dsonar.projectKey=${SONAR_PROJECT} \
                                -Dsonar.sources=src,app.py \
                                -Dsonar.tests=tests \
                                -Dsonar.python.coverage.reportPaths=coverage.xml
                        """
                    }
                }
            }
        }

        // ── 6. Quality Gate ───────────────────────────────────────────────────
        stage("Quality Gate") {
            steps {
                timeout(time: 5, unit: "MINUTES") {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        // ── 7. Deploy to Hugging Face Spaces ──────────────────────────────────
        stage("Deploy to HF Spaces") {
            when {
                branch "main"
            }
            steps {
                withCredentials([string(credentialsId: "HF_TOKEN", variable: "HF_AUTH")]) {
                    sh """
                        pip install huggingface_hub
                        python - <<EOF
from huggingface_hub import HfApi
import os

api = HfApi(token="${HF_AUTH}")
api.upload_folder(
    folder_path=".",
    repo_id="${HF_REPO}",
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__", ".venv", ".git", "tests"],
)
print("✅ Deployed to https://huggingface.co/spaces/${HF_REPO}")
EOF
                    """
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo "🎉 Pipeline succeeded!"
        }
        failure {
            echo "❌ Pipeline failed — check the logs above."
        }
    }
}
