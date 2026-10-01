"""Data models: (1) the Shared Project State that is persisted, (2) the JSON
shapes the LLM must return (validated by pydantic)."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


Mode = Literal["brief", "balanced", "detailed"]
Phase = Literal["intake", "awaiting_confirmation", "complete"]
# 'suggestion' is created ONLY by the system (a path the assistant proposed). It is never an owner fact.
Kind = Literal["fact", "assumption", "decision", "alternative", "gap", "suggestion"]
OwnerKind = Literal["fact", "assumption", "decision", "alternative", "gap"]
Stance = Literal["likes", "dislikes", "wants_changed", "unsure"]
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
    "wants_summary",
    "other",
]
Basis = Literal[
    "intake_process", "glossary", "path_explanation", "general_knowledge_unverified", "cannot_answer"
]

PREFIX = {"fact": "F", "assumption": "A", "decision": "D", "alternative": "L", "gap": "G", "suggestion": "S"}


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
    pros: list[str] = []  # suggestions only
    cons: list[str] = []  # suggestions only
    recommended: bool = False  # suggestions only: the assistant's own suggested starting point (never an owner view)
    reaction: str = ""  # owner's stance on a suggestion (Stance), "" = none yet
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
    owner_email: str = ""  # verified by a one-time code when the conversation was created; "" = no verified owner
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
    path_rounds: int = 0  # how many times paths were offered
    paths_turn: int = 0  # owner-turn number at which paths were last offered
    last_move: str = ""  # invite | suggest | revise | converse | recap | await
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
        pros: list[str] | None = None,
        cons: list[str] | None = None,
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
            pros=pros or [],
            cons=cons or [],
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
    kind: OwnerKind
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


class Reaction(BaseModel):
    target_id: str = Field(description="Id (S...) of an active system suggestion the owner is reacting to.")
    stance: Stance = Field(
        description="likes = owner favours it; dislikes = owner rejects it; wants_changed = owner wants it altered; unsure = owner is undecided."
    )
    source_quote: str = Field(description="Exact verbatim substring of the latest owner message showing the reaction.")


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
    reactions: list[Reaction] = Field(
        default_factory=list,
        description="Owner's reactions to active system suggestions (S ids). Empty if the owner did not react to any.",
    )


class QAnswer(BaseModel):
    question: str
    answer: str
    basis: Basis


class PathOption(BaseModel):
    name: str = Field(description="Short name for this possible path (3-8 words).")
    summary: str = Field(description="One or two sentences: what this path would mean for the owner's idea.")
    pros: list[str] = Field(description="2-3 short, qualitative advantages. No figures the owner did not give.")
    cons: list[str] = Field(description="2-3 short, qualitative drawbacks or risks. No figures the owner did not give.")


class ReplyDraft(BaseModel):
    answers: list[QAnswer] = Field(description="One entry per owner question; empty list if none.")
    restatement: str = Field(
        description="Only when the move is suggest/revise: the owner's idea restated clearly in 1-3 sentences from the state. Otherwise empty string."
    )
    paths: list[PathOption] = Field(
        description="Only when the move is suggest/revise: 2-3 genuinely different possible paths. Otherwise empty list."
    )
    lean_path: int = Field(
        default=0,
        description="Only when the move is suggest/revise: the 1-based number of the path you would suggest as a starting point for THIS owner, given what they said. 0 otherwise.",
    )
    lean_reason: str = Field(
        default="",
        description="Only when lean_path is set: 1-2 qualitative sentences on why that path is a sensible starting point, using only what the owner said. Never contains a question mark.",
    )
    giveback: str = Field(
        description="Only for converse/develop/invite/recap moves: substantive non-question content (reflection, a consideration, an update, a walk-through). Never contains a question mark. Otherwise empty string."
    )
    question: str = Field(description="The single question for this reply, or empty string if the move says not to ask.")


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


ReportBasis = Literal["owner_stated", "general_knowledge", "inference"]
ForceLevel = Literal["low", "moderate", "high", "cannot_assess"]


class SwotPoint(BaseModel):
    point: str = Field(description="One specific, plain-language point.")
    basis: ReportBasis = Field(
        description="owner_stated = comes from the hand-off (then source_ids is required); general_knowledge = general industry/strategy knowledge, qualitative only; inference = a reasoned implication of hand-off items (list them in source_ids)."
    )
    source_ids: list[str] = Field(
        description="Ids of hand-off items (F1, G2, ...) this point rests on. Only ids that exist in the hand-off. Empty only for general_knowledge."
    )


class Swot(BaseModel):
    strengths: list[SwotPoint] = []  # internal, positive - the owner's own resources
    weaknesses: list[SwotPoint] = []  # internal, negative - what the owner lacks or has not established
    opportunities: list[SwotPoint] = []  # external, favourable
    threats: list[SwotPoint] = []  # external, unfavourable


class ForceAssessment(BaseModel):
    intensity: ForceLevel = Field(description="Qualitative strength of this force; cannot_assess if neither the hand-off nor general knowledge supports a view.")
    reasoning: str = Field(description="2-4 sentences, qualitative, no invented figures or named companies.")
    basis: ReportBasis
    source_ids: list[str] = Field(description="Hand-off item ids this rests on; empty only for general_knowledge.")


class FiveForces(BaseModel):
    threat_of_new_entrants: ForceAssessment
    bargaining_power_of_suppliers: ForceAssessment
    bargaining_power_of_buyers: ForceAssessment
    threat_of_substitutes: ForceAssessment
    competitive_rivalry: ForceAssessment
    overall_takeaway: str = Field(description="2-3 sentences on what the five forces together mean for the attractiveness of this idea for THIS owner.")


class OpportunityReport(BaseModel):
    swot: Swot | None = None  # None only on reports saved before SWOT was added
    porters_five_forces: FiveForces | None = None  # None only on reports saved before this was added
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


class OpportunityReportOut(OpportunityReport):
    """What the LLM must return: SWOT and Five Forces are mandatory (the base class keeps them optional only
    so that reports saved earlier still load)."""

    swot: Swot
    porters_five_forces: FiveForces
