# Deployment & local development

## Docker (one command)

```bash
docker compose up --build
# Frontend: http://localhost:3000    API docs: http://localhost:8000/docs
```

## Local, with make (POSIX shells / Git Bash)

```bash
make install            # python -m venv .venv && pip install -e "backend[dev]"
make test               # backend pytest suite
make run                # uvicorn on :8000
make install-frontend   # npm install
make run-frontend       # next dev on :3000
```

## Local, without make (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e "backend[dev]"

# Backend (from the repo root)
.venv\Scripts\python -m uvicorn gb_battery.api.main:app --app-dir backend --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:3000 (proxies /api/* to :8000)
```

## Offline mode

Everything needed for a demo works without network access: choose the
**Bundled sample** or **Synthetic** source in the UI (Replay & Live page), or set
`GBB_OFFLINE=1` for the market-snapshot endpoints. The Elexon source requires
outbound HTTPS to `data.elexon.co.uk`.

## Checks

```bash
make test         # pytest (98 tests)
make lint         # ruff
make typecheck    # mypy
cd frontend && npx tsc --noEmit && npx next lint && npm run build
```
