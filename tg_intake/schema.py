"""Data models: (1) the Shared Project State that is persisted, (2) the JSON
shapes the LLM must return (validated by pydantic)."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


Mode = Literal["brief", "balanced", "detailed"]
Phase = Literal["intake", "awaiting_confirmation", "complete"]
Kind = Literal["fact", "assumption", "decision", "alternative", "gap"]
Param = Literal[
    "objective",
    "existing_business",
    "strengths",
    "capabilities",
    "customers",
    "feedstock_access",
    "geography_capital",
    "risk_appetite",
    "strategic_ambitions",
    "constraints",
    "scope",
]
GapType = Literal["", "unknown_to_owner", "not_provided", "ambiguous", "conflict"]
Intent = Literal[
    "providing_info",
    "asking_question",
    "correcting",
    "confirm_all",
    "declining_to_answer",
    "unclear",
    "other",
]
Basis = Literal[
    "intake_process", "glossary", "general_knowledge_unverified", "cannot_answer"
]

PREFIX = {"fact": "F", "assumption": "A", "decision": "D", "alternative": "L", "gap": "G"}


# ----------------------------------------------------------------------------
# Persisted state  (mirrors the "Shared Project State" box in the architecture:
# Facts / Assumptions / Decisions / Alternatives / Information Gaps / Evidence)
# ----------------------------------------------------------------------------
class Evidence(BaseModel):
    source_type: Literal["owner_statement", "system_note"] = "owner_statement"
    turn: int
    quote: str = ""
    at: str


class Item(BaseModel):
    id: str
    kind: Kind
    param: Param
    attribute: str = ""
    statement: str
    origin: Literal["owner", "system"] = "owner"
    status: Literal["active", "superseded", "withdrawn", "resolved"] = "active"
    confirmed: bool = False  # owner explicitly confirmed it
    confirmed_turn: int = 0
    gap_type: str = ""
    related_id: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    evidence: list[Evidence] = []
    created_turn: int = 0
    updated_at: str = ""


class OwnerQuestion(BaseModel):
    turn: int
    question: str
    answer_basis: str = ""
    answered: bool = False


class Turn(BaseModel):
    n: int
    role: Literal["owner", "assistant"]
    text: str
    at: str


class ProjectState(BaseModel):
    session_id: str
    owner_name: str = ""
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    mode: Mode = "balanced"
    mode_chosen: bool = False
    phase: Phase = "intake"
    turn: int = 0  # number of owner messages so far
    items: list[Item] = []
    owner_questions: list[OwnerQuestion] = []
    transcript: list[Turn] = []
    audit: list[dict] = []
    focus_attempts: dict[str, int] = {}
    last_focus: str = ""
    last_bot_question: str = ""
    next_id: int = 1

    # -- helpers ------------------------------------------------------------
    def active(self, kind: str | None = None, param: str | None = None) -> list[Item]:
        return [
            i
            for i in self.items
            if i.status == "active"
            and (kind is None or i.kind == kind)
            and (param is None or i.param == param)
        ]

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    def find_duplicate(self, kind: str, param: str, statement: str) -> Item | None:
        key = " ".join(statement.lower().split())
        for i in self.active(kind, param):
            if " ".join(i.statement.lower().split()) == key:
                return i
        return None

    def add_item(
        self,
        kind: str,
        param: str,
        statement: str,
        *,
        attribute: str = "",
        origin: str = "owner",
        quote: str = "",
        source: str = "owner_statement",
        gap_type: str = "",
        related_id: str = "",
        supersedes: str = "",
    ) -> Item:
        n = self.next_id
        self.next_id += 1
        item = Item(
            id=f"{PREFIX[kind]}{n}",
            kind=kind,
            param=param,
            attribute=attribute,
            statement=statement.strip(),
            origin=origin,
            gap_type=gap_type,
            related_id=related_id,
            supersedes=supersedes,
            evidence=[Evidence(source_type=source, turn=self.turn, quote=quote, at=now())],
            created_turn=self.turn,
            updated_at=now(),
        )
        self.items.append(item)
        return item


# ----------------------------------------------------------------------------
# LLM output shapes.  All fields are required so Gemini's structured output
# always returns them; "" / [] mean "nothing".
# ----------------------------------------------------------------------------
class Extraction(BaseModel):
    kind: Kind
    param: Param
    attribute: str = Field(description="1-3 word label, e.g. 'product', 'capacity', 'own funds'")
    statement: str = Field(
        description="Short neutral English sentence. Only facts and numbers that appear in source_quote."
    )
    source_quote: str = Field(
        description="Exact, verbatim, contiguous substring copied from the latest owner message."
    )
    gap_type: GapType = Field(description="Only when kind is 'gap'; otherwise empty string.")
    related_id: str = Field(
        description="Id of the existing item this conflicts with (for gap_type 'conflict'); otherwise empty string."
    )


class Correction(BaseModel):
    target_id: str = Field(description="Id of the existing active item being changed.")
    new_statement: str
    source_quote: str = Field(description="Exact verbatim substring of the latest owner message.")


class TurnAnalysis(BaseModel):
    owner_intent: Intent
    requested_mode: Literal["", "brief", "balanced", "detailed"] = Field(
        description="Set only if the owner explicitly asks for a different level of detail; else empty string."
    )
    extractions: list[Extraction]
    corrections: list[Correction]
    withdrawn_ids: list[str]
    resolved_ids: list[str]
    confirmed_ids: list[str]
    owner_questions: list[str] = Field(description="Every question the owner asked. Do not answer them here.")


class QAnswer(BaseModel):
    question: str
    answer: str
    basis: Basis


class ReplyDraft(BaseModel):
    answers: list[QAnswer] = Field(description="One entry per owner question; empty list if none.")
    acknowledgement: str = Field(description="At most one short sentence; empty string allowed.")
    question: str = Field(description="The next question for the owner, or empty string if the focus says not to ask.")


# ----------------------------------------------------------------------------
# Opportunity Evaluation Report (architecture boxes 2-3, collapsed into one
# evidence-grounded evaluation call): analysis across market, technology, raw
# materials, competition, CAPEX, operating economics, financing, risk and
# strategic fit; then assumptions, findings, recommendations and alternative
# opportunities. Generated from the intake hand-off, never from live intake.
# ----------------------------------------------------------------------------
AltType = Literal[
    "product", "technology_variant", "derivative", "upstream", "downstream", "adjacent_business"
]


class AlternativeOpportunity(BaseModel):
    name: str = Field(description="Short name for the alternative, e.g. 'Cold-pressed oil instead of raw export'.")
    type: AltType
    rationale: str = Field(description="Why this could be a better fit than the owner's original idea. 2-4 sentences.")
    fit_with_owner: str = Field(
        description="How this specifically uses the owner's own stated strengths, capabilities, feedstock, geography or customers - reference the items, do not invent new owner facts."
    )


class OpportunityReport(BaseModel):
    idea_summary: str = Field(description="One or two neutral sentences restating the owner's idea, from the hand-off only.")
    market_analysis: str
    technology_analysis: str
    raw_materials_analysis: str
    competition_analysis: str
    capex_analysis: str
    operating_economics_analysis: str
    financing_analysis: str
    risk_analysis: str
    strategic_fit_analysis: str
    assumptions: list[str] = Field(
        description="Every assumption this report relies on that the owner did not state directly - explicit and checkable, not buried in the prose above."
    )
    findings: list[str] = Field(description="The key evidence-based takeaways, short and specific, each traceable to hand-off items or a stated assumption.")
    recommendations: list[str] = Field(description="Concrete next steps or decisions for the owner, in priority order.")
    alternative_opportunities: list[AlternativeOpportunity] = Field(
        description="At least two genuinely different alternatives (products, technology variants, derivatives, upstream/downstream, or adjacent businesses) that better fit the owner's stated capabilities than the original idea."
    )
