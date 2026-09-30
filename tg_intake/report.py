"""Opportunity Evaluation Report generation - architecture boxes 2-3 (Strategy /
Technology / Estimate / Finance / Risk / Review agents and the Owner Review /
Decision Checkpoints) collapsed into a single evidence-grounded evaluation
call, matching box 3's "Opportunity Evaluation Report (consolidated analysis
& recommendations)".

Takes the intake hand-off (the confirmed Shared Project State) and asks
Gemini to evaluate the idea across market, technology, raw materials,
competition, CAPEX, operating economics, financing, risk and strategic fit;
to apply two standard strategic-management tools (SWOT and Porter's Five
Forces); to challenge the idea; and to propose alternatives that fit the
owner's own stated capabilities. See tg_intake/prompts.py:SYSTEM_REPORT for the
exact rules (most importantly: general business reasoning is allowed here,
unlike intake, but must never be presented as a verified fact).

Grounding is checked in code, not just requested: every hand-off id a SWOT
point or force cites must exist, owner-stated points must cite at least one,
and every SWOT quadrant must be filled. One retry is made with the exact
violations; if the report is still ungrounded it is rejected, not patched."""
from . import prompts
from .llm import LLM, LLMError
from .schema import OpportunityReport

FORCES = (
    ("threat_of_new_entrants", "threat of new entrants"),
    ("bargaining_power_of_suppliers", "bargaining power of suppliers"),
    ("bargaining_power_of_buyers", "bargaining power of buyers"),
    ("threat_of_substitutes", "threat of substitutes"),
    ("competitive_rivalry", "competitive rivalry"),
)
QUADRANTS = ("strengths", "weaknesses", "opportunities", "threats")


def handoff_ids(handoff: dict) -> set:
    """Ids of owner-side items in the hand-off (facts, assumptions, decisions, alternatives, gaps)."""
    state = handoff.get("shared_project_state", {})
    ids = set()
    for key in ("facts", "assumptions", "decisions", "alternatives", "information_gaps"):
        ids |= {row["id"] for row in state.get(key, [])}
    return ids


def _cite_problems(label: str, basis: str, source_ids: list, valid: set) -> list:
    problems = []
    unknown = [s for s in source_ids if s not in valid]
    if unknown:
        problems.append(f"{label} cites ids that are not in the hand-off: {unknown}")
    if basis in ("owner_stated", "inference") and not [s for s in source_ids if s in valid]:
        problems.append(f"{label} is marked '{basis}' but cites no hand-off item")
    return problems


def report_problems(handoff: dict, report: OpportunityReport) -> list:
    valid = handoff_ids(handoff)
    problems = []
    swot = report.swot
    if swot is None:
        problems.append("swot is missing")
    else:
        for q in QUADRANTS:
            points = getattr(swot, q)
            if not points:
                problems.append(f"swot.{q} is empty; give at least one specific point")
            for n, pt in enumerate(points, 1):
                if not pt.point.strip():
                    problems.append(f"swot.{q}[{n}] is blank")
                problems += _cite_problems(f"swot.{q}[{n}]", pt.basis, pt.source_ids, valid)
    five = report.porters_five_forces
    if five is None:
        problems.append("porters_five_forces is missing")
    else:
        for key, label in FORCES:
            f = getattr(five, key)
            if not f.reasoning.strip():
                problems.append(f"porters_five_forces.{key} has no reasoning")
            problems += _cite_problems(f"porters_five_forces.{key}", f.basis, f.source_ids, valid)
        if not five.overall_takeaway.strip():
            problems.append("porters_five_forces.overall_takeaway is empty")
    if len(report.alternative_opportunities) < 2:
        problems.append("alternative_opportunities must hold at least two alternatives")
    return problems


def generate_report(handoff: dict, llm: LLM) -> OpportunityReport:
    prompt = prompts.report_prompt(handoff)
    report = llm.evaluate(prompt)
    problems = report_problems(handoff, report)
    if problems:
        report = llm.evaluate(prompt + prompts.report_retry(problems))
        problems = report_problems(handoff, report)
    if problems:
        raise LLMError("the report could not be grounded in the hand-off: " + "; ".join(problems[:6]))
    return report
