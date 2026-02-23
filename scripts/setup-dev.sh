#!/usr/bin/env bash
set -euo pipefail

echo "=== LCM DocIntel Dev Environment Setup ==="

# 1. Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[OK] Created .env from .env.example"
else
    echo "[SKIP] .env already exists"
fi

# 2. Create uploads directory
mkdir -p uploads
echo "[OK] Created uploads directory"

# 3. Start Docker services
echo "[INFO] Starting Docker services..."
docker compose -f docker/docker-compose.dev.yml up -d
echo "[OK] Docker services started"

# 4. Wait for PostgreSQL
echo "[INFO] Waiting for PostgreSQL..."
until docker compose -f docker/docker-compose.dev.yml exec -T postgres pg_isready -U ragadmin -d docintel 2>/dev/null; do
    sleep 2
done
echo "[OK] PostgreSQL is ready"

# 5. Install Python dependencies
echo "[INFO] Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..
echo "[OK] Python dependencies installed"

# 6. Run database migrations
echo "[INFO] Running database migrations..."
cd backend
alembic upgrade head
cd ..
echo "[OK] Database migrations applied"

# 7. Verify Ollama models
echo "[INFO] Checking Ollama models..."
if command -v ollama &>/dev/null; then
    echo "  Ensure these models are pulled:"
    echo "    ollama pull qwen3:30b-a3b"
    echo "    ollama pull qwen3-embedding:0.6b"
else
    echo "  [WARN] Ollama not found. Install from https://ollama.ai"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the backend:  cd backend && uvicorn app.main:app --reload --port 8000"
echo "Start the frontend:  cd frontend && npm run dev"
echo "Health check:       curl http://localhost:8000/api/v1/health"
