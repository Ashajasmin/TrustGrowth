"""Opportunity Evaluation Report generation - architecture boxes 2-3 (Strategy /
Technology / Estimate / Finance / Risk / Review agents and the Owner Review /
Decision Checkpoints) collapsed into a single evidence-grounded evaluation
call, matching box 3's "Opportunity Evaluation Report (consolidated analysis
& recommendations)".

Takes the intake hand-off (the confirmed Shared Project State) and asks
Gemini to evaluate the idea across market, technology, raw materials,
competition, CAPEX, operating economics, financing, risk and strategic fit;
challenge the idea; and propose alternatives that fit the owner's own stated
capabilities. See tg_intake/prompts.py:SYSTEM_REPORT for the exact rules
(most importantly: general business reasoning is allowed here, unlike
intake, but must never be presented as a verified fact)."""
from .llm import LLM
from .prompts import report_prompt
from .schema import OpportunityReport


def generate_report(handoff: dict, llm: LLM) -> OpportunityReport:
    return llm.evaluate(report_prompt(handoff))
