# MangoCare AI (MVP)
Preliminary, image-based **mango leaf** disease screening for educational use. React + Vite + TypeScript + Tailwind frontend, FastAPI backend, SQLite metadata, ReportLab PDF reports.
Not a laboratory diagnosis. See `MODEL_CARD.md` for model provenance and limitations.

## What works
Upload → validation (decode/size/mango check) → AA-ENet classification → abstention policy → knowledge-base guidance → SQLite history → PDF report. Uploaded images are never stored.

## Setup (needs Python 3.10–3.12, Node 18+, internet on first run)
```bash
# macOS / Linux
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cd ../frontend && npm install && cd ..
./run_unix.sh
```
```powershell
# Windows PowerShell
cd backend; python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; cd ..\frontend; npm install; cd ..
.\run_windows.ps1
```
Open http://localhost:5173 (API docs: http://localhost:8000/docs). **First backend start downloads CLIP (~600 MB) from Hugging Face**; until it loads, `/api/health` says "degraded" and `/api/analyze` returns 503 (no fake results).
Config: see `.env.example` (all optional). Tests: `cd backend && python -m pytest -q`.

## Result meanings
- **Healthy appearance**: no supported disease matched; NOT a guarantee.
- **Possible disease**: top class passed the confidence policy; confirm with an expert.
- **Uncertain**: low confidence; retake the photo.
- **Invalid image** (HTTP 422): no mango detected or unreadable image; nothing stored.
- **Service unavailable** (HTTP 503): a model failed to load.

## API
`GET /api/health` · `GET /api/model-info` · `POST /api/analyze` (multipart `file`) · `GET /api/history` · `GET|DELETE /api/history/{id}` · `GET /api/diseases[?q=]` · `GET /api/diseases/{id}` · `GET /api/reports/{id}`
```bash
curl -F "file=@leaf.jpg" http://localhost:8000/api/analyze
```

## Known MVP gaps (honest status)
- Real CLIP validator and real-image accuracy were **not verified** by the author (sandbox had no Hugging Face access). Backend tests: 18 pass, 1 skipped (real CLIP). Mock-validator tests only check plumbing.
- No model evaluation metrics exist; none are shown.
- Disease-library references are unverified and marked so; two scientific names are marked "(verify)".
- Not built: demo mode, Docker, frontend unit tests, evaluation script, PDF photo embedding (images aren't retained), multi-page routing/charts.
