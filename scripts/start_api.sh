#!/bin/sh
# Cloud-friendly API start script (Render, Railway, etc.)
PORT="${PORT:-8000}"
exec uvicorn app.api.main:app --host 0.0.0.0 --port "$PORT"
