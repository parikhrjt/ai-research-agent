.PHONY: help install test lint run-api run-ui docker-up docker-down seed pdf

help:
	@echo "AI Research Agent — available targets:"
	@echo "  install     Install dependencies in .venv"
	@echo "  test        Run unit tests"
	@echo "  lint        Run ruff linter"
	@echo "  run-api     Start FastAPI dev server"
	@echo "  run-ui      Start Streamlit frontend"
	@echo "  docker-up   Launch full Docker Compose stack"
	@echo "  docker-down Stop Docker Compose stack"
	@echo "  seed        Ingest sample documents into vector store"
	@echo "  pdf         Generate sample PDF document"

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt fpdf2

test:
	PYTHONPATH=. .venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check app tests

run-api:
	PYTHONPATH=. .venv/bin/uvicorn app.api.main:app --reload --port 8000

run-ui:
	PYTHONPATH=. API_BASE_URL=http://localhost:8000 .venv/bin/streamlit run app/ui/streamlit_app.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

seed:
	PYTHONPATH=. .venv/bin/python scripts/seed_samples.py

pdf:
	.venv/bin/python scripts/generate_sample_pdf.py
