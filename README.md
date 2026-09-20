# TG Opportunity Finder - Stage 1: Intake Conversation Bot

Implements box **1 (Intake Conversation)** of the revised architecture and hands off to the
**Shared Project State** (box 2). Model: `gemini-3.1-flash-lite` (change with `TG_MODEL` in `.env`).

## Run
```
python -m venv .venv && .venv\Scripts\activate
pip install -U -r requirements.txt
copy .env.example .env        # then put your GEMINI_API_KEY in .env
python -m pytest -q           # 14 tests, no API key needed (scripted fake LLM)
python cli.py --owner "Mr. Goyal"        # terminal chat  (/state /mode /export /quit)
uvicorn server:app --reload              # web chat at http://127.0.0.1:8000
```

## How it maps to the architecture
| Architecture | Where |
|---|---|
| Adaptive Conversation: Brief / Balanced / Detailed, pyramid, minimum info first | `framework.py` (`MODES`, level 1 -> 2 -> 3 order, `plan_next`) |
| Clarify & Deepen: ask, confirm, capture intent & scope | ambiguity/conflict gaps trigger a clarifying question; recap + confirm step; scope question |
| Intake Complete: objective clear, scope defined, key constraints & gaps | completeness computed in code (`param_status`), never declared by the LLM |
| Move to Shared Project State | `data/handoff/<session>.json` (facts, assumptions, decisions, alternatives, information_gaps, evidence) |
| Owner deliberation loop (provide info, ask, clarify, confirm) | owner questions answered in-flow; corrections supersede (history kept); `confirm` step |
| 8 company parameters | strengths, existing business/employment, capabilities, customers, feedstock access, geography+capital, risk appetite, strategic ambitions (`SPECS`) |

## Memory
`data/sessions/<id>.json` holds everything: items with evidence quotes, owner questions, full transcript, audit log.
Resume with `python cli.py --resume <id>`.

## Anti-hallucination design
1. Every stored item needs an **exact quote** from the owner's message - checked in code (`guards.py`); otherwise rejected and logged.
2. **Numbers** in a stored statement must appear in the owner's quote; numbers in the bot's replies must come from the owner or the state, else regenerated/redacted.
3. Everything is labelled **owner-stated, not verified**. The LLM cannot mark anything verified.
4. Owner questions are answered only from the process text / glossary; anything needing market data, prices, rules or advice is answered "cannot answer at intake" and queued for the analysis stages. General-knowledge answers get an "unverified" note.
5. "Don't know" / skip becomes an explicit **information gap**; after repeated non-answers a `not_provided` gap is recorded.
6. The recap is generated from stored items, not by the LLM. The only system-made item is the default scope, flagged as a system assumption for the owner to confirm.

## Not done / decide with TrustGrowth
- Not yet run against the live Gemini API (tested with a scripted fake LLM only) - expect to tune prompts after the first real conversations.
- Owner conversations are sent to Google's Gemini API. The brief says no client-identifiable information may leave the agreed environment - confirm this is acceptable (or use anonymised test data).
- Background research (box on the right) and the analysis agents are out of scope for this bot.
