# TG Opportunity Finder — Run Guide and Working Document

Project folder: `D:\asha\TrustGrowth`
Stage implemented: **Box 1 — Intake Conversation** of the revised architecture, plus the
consolidated **Opportunity Evaluation Report** that boxes 2–3 produce.
Model: **Gemini** (`TG_MODEL`, default `gemini-3.1-flash-lite`).

---

## Part A — Step-by-step commands to run it

Open **PowerShell** (or Command Prompt) and run these in order.

### 1. Go to the project
```
D:
cd D:\asha\TrustGrowth
```

### 2. Activate the virtual environment
The `.venv` folder already exists in this project.
```
.venv\Scripts\activate
```
Your prompt should now start with `(.venv)`.

> If PowerShell blocks the script with an execution-policy error, run this once in the same window:
> ```
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> If the environment does not exist at all, create it first:
> ```
> python -m venv .venv
> .venv\Scripts\activate
> ```

### 3. Install / refresh the dependencies
```
pip install -U -r requirements.txt
```
Installs `google-genai`, `pydantic`, `python-dotenv`, `fastapi`, `uvicorn`, `pytest`.

### 4. Check the API key
`.env` already exists in the folder. Confirm it contains your Gemini key:
```
type .env
```
It must have a line `GEMINI_API_KEY=...` (and optionally `TG_MODEL=...`).
If `.env` is missing, create it from the template and then edit it:
```
copy .env.example .env
notepad .env
```

### 5. Run the tests (optional but recommended — no API key needed)
```
python -m pytest -q
```
These use a scripted fake LLM, so they pass without calling Gemini.

### 6. Start the web app
```
uvicorn server:app --reload
```
Leave this window running. Open a browser at:
```
http://127.0.0.1:8000
```
Stop the server with `Ctrl + C`.

> **Important:** hard-refresh the browser the first time (`Ctrl + F5`) so the old cached
> `index.html` is not used.

### 7. Use it in the browser
1. Click **New intake conversation**, optionally type the owner's name, click **Start**.
2. Answer the assistant's questions in the chat box. Use **Brief / Balanced / Detailed**
   at the top to change how deeply it probes.
3. Watch the right-hand panel: **Intake** (the 8 parameters and their status),
   **Project state** (every captured fact/assumption/gap), **Pipeline** (where you are).
4. When it shows the recap, reply **confirm** (or click the Confirm chip).
5. Click **Generate evaluation report**. Wait — this is one long Gemini call.
6. In the report view use **Print / PDF** or **Download JSON**.

### 8. (Alternative) Run it in the terminal instead of the browser
```
python cli.py --owner "Mr. Goyal"
python cli.py --list
python cli.py --resume <session_id>
```
In-chat commands: `/state`, `/mode brief|balanced|detailed`, `/export`, `/help`, `/quit`.
The CLI covers the intake only; the report view is browser-only.

### 9. Where the output lands
```
data\sessions\<session_id>.json   full state + transcript + audit log
data\handoff\<session_id>.json    confirmed hand-off for the analysis stages
data\reports\<session_id>.json    the generated Opportunity Evaluation Report
```

---

## Part B — How the project works

### B.1 Folder map

| Path | What it is |
|---|---|
| `server.py` | FastAPI app. Serves the single HTML page and the JSON API. |
| `cli.py` | Terminal version of the same intake conversation. |
| `static/index.html` | **The entire frontend** — one self-contained file (HTML + CSS + JS, no build step, no framework). |
| `tg_intake/config.py` | Reads `.env`: model, API key, data dir, thinking level, history window. |
| `tg_intake/schema.py` | All data models: the persisted project state and the JSON shapes Gemini must return. |
| `tg_intake/framework.py` | The 8 parameters, the 3 depth modes, the pyramid order, coverage, and "what to ask next". Plain Python — not the LLM. |
| `tg_intake/engine.py` | The turn loop: analyze → validate → plan → respond → save. |
| `tg_intake/guards.py` | Anti-hallucination checks (quote must exist, numbers must be the owner's). |
| `tg_intake/prompts.py` | System instructions and per-turn prompt assembly. |
| `tg_intake/llm.py` | Gemini wrapper. Structured output against a pydantic schema, one retry. |
| `tg_intake/report.py` | Turns the confirmed hand-off into the Opportunity Evaluation Report. |
| `tg_intake/store.py` | JSON persistence (atomic writes) + hand-off builder. |
| `tests/test_engine.py` | Engine tests driven by a scripted fake LLM. |
| `data/` | `sessions/`, `handoff/`, `reports/`. |

### B.2 What happens on one message

1. **Analyze (Gemini).** The owner's message plus recent history is sent, and the model must
   return a `TurnAnalysis`: intent, extractions, corrections, withdrawn/resolved/confirmed ids,
   and any questions the owner asked.
2. **Validate and apply (Python).** Every extraction must carry an exact verbatim quote from
   the owner's message (`guards.quote_in_message`), and every number in a stored statement must
   also appear in that quote (`guards.numbers_supported`). Anything failing is rejected and
   written to the audit log — it never enters the state.
3. **Plan next (Python).** `framework.plan_next` decides the next focus: first clear up
   conflicts and ambiguities, then walk the parameters in pyramid order (level 1 essentials →
   level 2 what you bring → level 3 appetite and ambition), then constraints, then scope,
   then the recap. Completeness is computed in code — the LLM is never allowed to declare
   the intake finished.
4. **Respond (Gemini).** The model only writes the wording. Numbers in the draft that the owner
   never gave are regenerated, and if they persist, the offending sentences are redacted.
5. **Save.** The whole state, transcript and audit trail is written atomically to
   `data/sessions/<id>.json`.

On **confirm**, every active item is marked confirmed, the phase becomes `complete`, and
`data/handoff/<id>.json` is exported. The report is generated from that hand-off only.

### B.3 The 8 input parameters (from the brief)

Asked in pyramid order, not brief order. The objective (the idea itself) is captured first.

| # | Parameter | Key in code |
|---|---|---|
| 1 | Strengths | `strengths` |
| 2 | Existing business or employment | `existing_business` |
| 3 | Capabilities | `capabilities` |
| 4 | Customers | `customers` |
| 5 | Feedstock access | `feedstock_access` |
| 6 | Geography and capital | `geography_capital` |
| 7 | Risk appetite | `risk_appetite` |
| 8 | Strategic ambitions | `strategic_ambitions` |

Plus `objective`, `constraints` and `scope`.

Each one is reported as **Captured**, **In progress**, **Unknown** (the owner said they do not
know, or it was never provided) or **Not asked yet**. "Unknown" is stored as an explicit
information gap — never guessed.

### B.4 The report

`POST /api/session/{id}/report` sends the confirmed hand-off to Gemini and validates the answer
against the `OpportunityReport` schema, which forces:

- **Analysis** across nine dimensions: market, technology, raw materials, competition, CAPEX,
  operating economics, financing, risk, strategic fit.
- **Assumptions** — everything the report leans on that the owner did not state.
- **Findings** — the evidence-based takeaways.
- **Recommendations** — concrete next steps, in priority order.
- **Alternative opportunities** — at least two genuinely different options (different product,
  technology variant, derivative, upstream, downstream, adjacent business), each with a rationale
  and an explicit "fit with the owner's stated capabilities".

That last block is the "challenge the original idea" requirement from the brief.

### B.5 The API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Serves `static/index.html`. |
| GET | `/api/sessions` | List saved conversations. |
| POST | `/api/session` | Start one (`owner_name` optional). |
| GET | `/api/session/{id}` | Full transcript + state view. |
| POST | `/api/session/{id}/message` | Send an owner message, get the reply. |
| POST | `/api/session/{id}/mode` | brief / balanced / detailed. |
| GET | `/api/session/{id}/handoff` | The hand-off JSON. |
| POST | `/api/session/{id}/report` | Generate the report (400 unless the intake is confirmed). |
| GET | `/api/session/{id}/report` | Fetch an already-generated report (404 if none). |

---

## Part C — The frontend: before and after

Only one file was changed: **`static/index.html`**. No backend file was touched, and no new
dependency was added. The previous version is still in git history.

### Before

- A working chat page: sidebar of sessions, chat area, mode switch, and a right panel with
  Progress / Captured / Roadmap tabs.
- It called five endpoints: `/api/sessions`, `/api/session`, `/api/session/{id}`,
  `/api/session/{id}/message`, `/api/session/{id}/mode`, plus the hand-off download.
- **The report endpoints were never called.** The stylesheet already contained rules for a
  report view (`.reportView`, `.rsec`, `.altcard`, `.rlist`, …), but there was no matching
  markup and no JavaScript for it — dead CSS.
- The Roadmap tab told the user that only the intake was built and that the report "is not part
  of this version", which was no longer true of the backend.
- So the product requirement "the user receives a comprehensive, evidence-based report" had no
  route through the UI. It could only be triggered by hand, e.g. with curl.

### After

Everything above is kept and the gap is closed:

1. **Report view added.** A full-screen panel inside the chat area with a real template:
   idea summary → 1 Analysis (the nine dimensions, numbered, in the brief's order) →
   2 Assumptions → 3 Findings → 4 Recommendations (numbered) → 5 Alternative opportunities
   (cards with a type badge, rationale, and fit-with-owner).
2. **Report wiring.** `POST` to generate, `GET` to reload a report that already exists when an
   old session is reopened. A loading state during the call, and errors from the server
   (including the 400 "confirm the intake first" and the 502 LLM failure) shown verbatim with a
   Try again button instead of failing silently.
3. **Report export.** Download JSON, and a print stylesheet so `Print / PDF` produces a clean
   document with the chrome hidden.
4. **The 8 parameters made explicit.** The right panel numbers them 1–8 in the brief's order,
   shows the objective separately, and highlights the parameter currently being asked using the
   `asking` field the API already returned but the old page under-used.
5. **Project state tab.** Captured items grouped by parameter and labelled Fact / Assumption /
   Decision / Alternative / Information gap, with system-made assumptions flagged for confirmation.
6. **Pipeline tab corrected.** It now reflects the real stages — intake, shared project state,
   evaluation, challenge and alternatives, report, investment decision — and marks which are done.
7. **Completion card.** When the intake is confirmed, the chat shows "Generate evaluation report"
   and "Download hand-off" instead of only the hand-off download.
8. **Smaller fixes.** Server error messages are read from FastAPI's `detail` field instead of a
   generic "Server error"; `Esc` closes the report and the dialog; the report button appears in
   the header once the phase is `complete`; refreshed visual styling (Claude/ChatGPT-style
   layout, light and dark mode).

### What was deliberately not changed

- No backend logic, no prompt, no schema, no guard.
- No new libraries — still a single static HTML file, which is all `server.py` serves
  (`FileResponse(static/index.html)`); there is no `StaticFiles` mount, so a multi-file
  frontend would not have loaded.
- The intake conversation flow itself is unchanged; the engine still decides every question.

---

## Part D — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No API key found. Put GEMINI_API_KEY=... in the .env file.` | `.env` missing or the key line is empty. |
| Page looks like the old one | Browser cache — press `Ctrl + F5`. |
| "Confirm the intake before generating…" | The intake phase is not `complete` yet; reply **confirm** to the recap first. |
| "Report generation failed" (502) | The Gemini call failed or returned invalid JSON. Click Try again; check the key, the model name in `.env`, and your internet connection. |
| 404 Session not found | The session JSON was deleted from `data\sessions\`. Start a new conversation. |
| `uvicorn` not recognised | The virtual environment is not activated — run `.venv\Scripts\activate`. |
| Assistant replies "Sorry - I had trouble processing that" | The analyze call failed; nothing was changed in your state. Send the message again. |
