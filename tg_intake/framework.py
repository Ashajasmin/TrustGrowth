"""Intake framework: the 8 company-defined input parameters, the three depth
modes (Brief / Balanced / Detailed), the pyramid ordering, coverage and the
"what do we ask next" planner. All of this is plain code - NOT left to the LLM -
so completeness is decided deterministically."""
from dataclasses import dataclass

from .schema import ProjectState


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    level: int  # pyramid level: 1 = essentials first, 2 = who/what you bring, 3 = appetite & ambition
    question: str  # default question (used if the LLM gives none)
    checklist: tuple  # aspects to probe in Detailed mode (prompts, not facts)


SPECS = (
    ParamSpec("objective", "Objective (idea, scale, reason)", 1,
              "What product or business idea are you considering, and roughly what scale do you have in mind?",
              ("product or idea", "intended plant size / capacity", "why this idea")),
    ParamSpec("existing_business", "Existing business or employment", 1,
              "What business or job do you have today?",
              ("current business or employment", "sector", "your role", "size of current operations")),
    ParamSpec("geography_capital", "Geography and capital", 1,
              "Where would you like to set this up, and how much capital can you commit - your own money versus a loan?",
              ("location(s) or land", "total capital available", "own funds", "borrowed funds")),
    ParamSpec("strengths", "Strengths", 2,
              "What do you see as your biggest strengths for this venture (for example contacts, distribution, money, land, experience)?",
              ("contacts / relationships", "distribution", "financial strength", "land or location", "experience")),
    ParamSpec("capabilities", "Capabilities", 2,
              "What can you and your team already do in-house - technical, operations, management?",
              ("technical know-how", "operations / production experience", "management team", "existing facilities or equipment")),
    ParamSpec("customers", "Customers", 2,
              "Who would buy this product - do you have customers in mind, and have you spoken to any of them?",
              ("target customers", "existing relationships", "whether they have been approached", "any agreement or commitment")),
    ParamSpec("feedstock_access", "Feedstock (raw material) access", 2,
              "What raw materials (feedstock) would you need, and how would you get access to them?",
              ("main raw materials", "suppliers or sources", "own / captive access", "whether supply is confirmed")),
    ParamSpec("risk_appetite", "Risk appetite", 3,
              "How much risk are you comfortable taking on this investment?",
              ("overall comfort with risk", "loss you could tolerate", "all-at-once versus staged investment")),
    ParamSpec("strategic_ambitions", "Strategic ambitions", 3,
              "What do you want this venture to do for your business in the longer run?",
              ("growth aims", "time horizon", "diversification or new industries", "future scale-up")),
)
SPEC_BY_KEY = {s.key: s for s in SPECS}
ORDER = [s.key for s in sorted(SPECS, key=lambda s: s.level)]  # stable: keeps order inside a level
EXTRA_LABELS = {"constraints": "Key constraints", "scope": "Scope of analysis"}

CONSTRAINTS_QUESTION = (
    "Are there any hard constraints I should know about - for example timing, land, regulation, partners or a budget ceiling? "
    "\"None\" is a fine answer."
)
SCOPE_QUESTION = (
    "What would you like from the analysis: should it only evaluate your idea, or also challenge it and suggest better alternatives that fit you?"
)
DEFAULT_SCOPE = (
    "Scope assumed from the TrustGrowth process: evaluate the owner's idea and also look for better alternatives. "
    "Not yet stated by the owner."
)


@dataclass(frozen=True)
class ModeConfig:
    min_facts: int  # facts needed for a parameter to count as covered
    max_attempts: int  # owner replies on a topic before we accept a gap and move on
    ask_constraints: bool
    ask_scope: bool
    clarify_ambiguous: bool
    style: str


MODES = {
    "brief": ModeConfig(1, 2, False, False, False,
        "BRIEF: minimum information first. Accept short answers, one question per turn, do not probe for detail."),
    "balanced": ModeConfig(1, 3, True, True, True,
        "BALANCED: one main question per turn; follow up once when an answer is vague or misses something important."),
    "detailed": ModeConfig(2, 3, True, True, True,
        "DETAILED: probe each topic in depth using its checklist; ask for specifics (quantities, units, names, dates)."),
}


@dataclass
class Focus:
    kind: str  # param | clarify | constraints | scope | recap | await_confirm
    key: str
    param: str = ""
    item_id: str = ""


def label_of(key: str) -> str:
    return SPEC_BY_KEY[key].label if key in SPEC_BY_KEY else EXTRA_LABELS.get(key, key)


def param_status(st: ProjectState, key: str) -> str:
    cfg = MODES[st.mode]
    if len(st.active("fact", key)) >= cfg.min_facts:
        return "covered"
    if any(g.gap_type in ("unknown_to_owner", "not_provided") for g in st.active("gap", key)):
        return "gap"
    return "partial" if st.active(param=key) else "empty"


def coverage(st: ProjectState) -> dict:
    return {k: param_status(st, k) for k in ORDER}


def plan_next(st: ProjectState) -> Focus:
    """Decide what the conversation should do next (pyramid: level 1 -> 2 -> 3)."""
    cfg = MODES[st.mode]
    if st.phase == "awaiting_confirmation":
        return Focus("await_confirm", "await_confirm")

    # 1. things the owner said that are unclear or contradictory
    for g in st.active("gap"):
        wanted = g.gap_type == "conflict" or (g.gap_type == "ambiguous" and cfg.clarify_ambiguous)
        if wanted and st.focus_attempts.get(f"clarify:{g.id}", 0) < 2:
            return Focus("clarify", f"clarify:{g.id}", param=g.param, item_id=g.id)

    # 2. the eight parameters, broad to deep
    for key in ORDER:
        status = param_status(st, key)
        if status in ("covered", "gap"):
            continue
        if st.focus_attempts.get(key, 0) >= cfg.max_attempts:
            if status == "empty":  # honest record of what we could not get
                st.add_item("gap", key, f"No information provided on: {label_of(key)}.",
                            origin="system", source="system_note", gap_type="not_provided")
            continue
        return Focus("param", key, param=key)

    # 3. key constraints
    if cfg.ask_constraints and not st.active(param="constraints") and st.focus_attempts.get("constraints", 0) < 1:
        return Focus("constraints", "constraints", param="constraints")

    # 4. scope
    if not st.active("decision", "scope"):
        if cfg.ask_scope and st.focus_attempts.get("scope", 0) < 1:
            return Focus("scope", "scope", param="scope")
        if not st.active("assumption", "scope"):
            st.add_item("assumption", "scope", DEFAULT_SCOPE, origin="system", source="system_note")

    return Focus("recap", "recap")


def default_question(focus: Focus, st: ProjectState) -> str:
    if focus.kind == "param":
        return SPEC_BY_KEY[focus.key].question
    if focus.kind == "clarify":
        item = st.get(focus.item_id)
        return f"Could you clarify this for me: {item.statement if item else 'your last point'}?"
    if focus.kind == "constraints":
        return CONSTRAINTS_QUESTION
    if focus.kind == "scope":
        return SCOPE_QUESTION
    return ""


# ----------------------------------------------------------------------------
# Recap (built from stored items only - the LLM does not write it)
# ----------------------------------------------------------------------------
_GAP_LABEL = {
    "unknown_to_owner": "Not known yet",
    "not_provided": "Not provided",
    "ambiguous": "Needs clarification",
    "conflict": "[!] Conflict",
}
_KIND_TAG = {"fact": "", "assumption": "Assumption: ", "decision": "Decision: ", "alternative": "Alternative considered: "}


def _line(i) -> str:
    if i.kind == "gap":
        return f"- {_GAP_LABEL.get(i.gap_type, 'Gap')}: {i.statement} [{i.id}]"
    extra = " _(assumed by the system - please confirm)_" if i.origin == "system" else ""
    return f"- {_KIND_TAG[i.kind]}{i.statement} [{i.id}]{extra}"


def render_recap(st: ProjectState) -> str:
    out = [
        "Here is what I have captured. It is based only on what you told me and has **not** been independently verified yet:",
        "",
    ]
    for key in [s.key for s in SPECS] + ["constraints", "scope"]:
        items = st.active(param=key)
        out.append(f"**{label_of(key)}**")
        out.extend(_line(i) for i in items) if items else out.append("- (nothing captured)")
        out.append("")
    out.append('Please reply **confirm** if this is right, or tell me what to change or add.')
    return "\n".join(out)
