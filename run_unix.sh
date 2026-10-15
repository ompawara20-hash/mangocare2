#!/usr/bin/env bash
# Starts backend (8000) and frontend (5173). Run setup first (see README).
set -e; cd "$(dirname "$0")"
(cd backend && python3 -m uvicorn app.main:app --port 8000) & B=$!
trap "kill $B" EXIT
cd frontend && npm run dev
