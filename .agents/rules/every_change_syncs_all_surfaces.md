# Rule: Every change reaches ALL surfaces (Streamlit + FastAPI + Web UI) — no reminder needed

The user should never have to ask for this. Any feature, fix, removal, label, default or number changed in
one place MUST be applied in the same task to every surface below, then verified.

## The surfaces
| Surface | Files | Check at |
|---|---|---|
| Shared logic (single source of truth) | `refillcare/**`, `database/models.py`, `reminder/**` | unit tests |
| Streamlit pharmacist app | `app_refillcare.py` | headless `streamlit.testing.v1.AppTest` |
| FastAPI | `api/main.py` (routes), `api/schemas.py` (request/response fields), `api/services.py` | `http://127.0.0.1:8000/docs` (OpenAPI) |
| Web UI served by FastAPI | `frontend/index.html`, `frontend/js/api.js`, `frontend/js/app.js`, `frontend/css/styles.css` | `http://127.0.0.1:8000/` |
| Docs / config | `implementation_plan.md`, `README.md`, `.env.example`, `docs/*` | — |

## Mandatory checklist for every change
1. **Logic lives once** in a shared module (`refillcare/...`). Streamlit and FastAPI both call it — never re-implement a
   calculation in a UI, never let Streamlit call FastAPI or FastAPI import Streamlit.
2. **API**: add/modify the route in `api/main.py`, the Pydantic fields in `api/schemas.py` (with `description=`), and the
   tag, so `/docs` shows it. Removed features: remove or deprecate their routes/fields.
3. **Web UI**: add the call in `frontend/js/api.js`, render it in `frontend/js/app.js` + `frontend/index.html`, style in
   `styles.css`. Same labels, defaults (e.g. date = today), filters, confirmations and hidden/removed sections as Streamlit.
   Every request sends `X-Store-Id` (from the page URL `?store_id=`) so the web UI follows the same store.
4. **No hard-coded metrics, dates or months** in any surface (`'2026-10'`, fixed counts, typed-in accuracy). Compute from the store's DB.
5. **Multi-store**: caches keyed by store (`store_cache_data` in Streamlit), no fallback to another store's files.
6. **Verify before reporting done**:
   - `pytest -q` (full suite) passes;
   - Streamlit AppTest: 0 exceptions;
   - start `uvicorn api.main:app --port 8000`, check every new/changed route in `/openapi.json` and call it;
   - load `/` and the JS files (no 404s; `node --check frontend/js/*.js` when node is available).
   - confirm the RUNNING server on :8000 returns the new fields (Windows `--reload` can leave an orphan
     `multiprocessing` worker serving old code on the port: stop it, restart uvicorn, re-check `/openapi.json`).
   - every number shown in more than one tab (customers, sales lines, dates, reminders) must come from ONE
     shared function (`history_store.store_summary`, `channel_counts`, `overview_kpis`) - never recount in a tab.
   - web UI tab switches must fetch fresh data (no stale client cache after uploads / adds / sends).
7. **Report** in the final message a small table: change → Streamlit / API route / Web UI / tests.

## Safety (unchanged)
Never send real WhatsApp messages, never write to Supabase, never expose `.env` secrets, never `git push`, back up
`enterprise.db` before changing production model/DB state.
