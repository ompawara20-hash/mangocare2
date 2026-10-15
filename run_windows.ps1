# Starts backend (8000) and frontend (5173). Run setup first (see README).
Set-Location $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$PSScriptRoot\backend'; python -m uvicorn app.main:app --port 8000"
Set-Location frontend; npm run dev
