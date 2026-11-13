# RefillCare — standing instructions for Claude

**Every change must reach ALL surfaces in the same task, without the user asking:**
shared logic (`refillcare/**`) → Streamlit (`app_refillcare.py`) → FastAPI (`api/main.py`, `api/schemas.py`,
`api/services.py`, visible in `http://127.0.0.1:8000/docs`) → Web UI (`frontend/index.html`, `frontend/js/api.js`,
`frontend/js/app.js`, `frontend/css/styles.css`, served at `http://127.0.0.1:8000/`) → docs (`implementation_plan.md`, `README.md`, `.env.example`).

Full checklist: `.agents/rules/every_change_syncs_all_surfaces.md`. Also follow the other files in `.agents/rules/`.

Verify before saying done: full `pytest -q`, Streamlit AppTest with 0 exceptions, run uvicorn on :8000 and check
`/openapi.json` + `/` + changed routes. End with a change → Streamlit / API / Web UI / tests table.

Never: send real WhatsApp messages, write to Supabase, print `.env` secrets, `git push`, or commit (the user commits).
Pharmacist-facing app: no login/store picker (Mediastra handles logins; store comes from `?store_id=` / `X-Store-Id`).
