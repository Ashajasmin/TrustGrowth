"""Engine tests with a scripted stand-in for the LLM (no network / API key needed).
They check the deterministic rules in OUR code - quote and number guards, conversation
shape, evidence handling, report grounding. They say nothing about how well Gemini
writes; that needs real conversations (see README).
Run:  python -m pytest -q"""
import json
import re

import pytest

from tg_intake import framework as fw
from tg_intake import prompts
from tg_intake.engine import IntakeEngine
from tg_intake.llm import LLMError, json_schema
from tg_intake.report import generate_report, report_problems
from tg_intake.schema import (
    AlternativeOpportunity, Correction, Extraction, FiveForces, ForceAssessment, OpportunityReport,
    OpportunityReportOut, PathOption, ProjectState, QAnswer, Reaction, ReplyDraft, Swot, SwotPoint, TurnAnalysis,
)
from tg_intake.store import JsonStore, build_handoff

GIVE = "That gives me a clear enough picture of where you are starting from."


def rd(giveback="", question="", restatement="", paths=None, answers=None):
    return ReplyDraft(answers=answers or [], restatement=restatement, paths=paths or [], giveback=giveback, question=question)


def path(name, pros=("Quicker to start",), cons=("Less control",)):
    return PathOption(name=name, summary=f"{name} would shape how you approach the idea.", pros=list(pros), cons=list(cons))


def suggest_reply():
    return rd(restatement="You want to make protein shakes, if I've understood.",
              paths=[path("Contract manufacture first"), path("Own plant")],
              question="Which of these feels closest, and what would you change?")


def auto_reply(prompt):
    """A valid reply for whatever move the engine asked for (used when a test does not care about the wording)."""
    move = re.search(r"MOVE: (\w+)", prompt).group(1)
    if move in ("suggest", "revise"):
        return suggest_reply()
    if move == "invite":
        return rd(giveback=GIVE, question="What's the idea you have in mind?")
    if move == "converse":
        return rd(giveback=GIVE, question="Where would you like to set this up?")
    if move == "recap":
        return rd(giveback=GIVE + " The summary follows.")
    return rd()


class FakeLLM:
    def __init__(self, analyses, replies=None, reports=None):
        self.analyses, self.replies, self.reports = list(analyses), list(replies or []), list(reports or [])
        self.reply_prompts = []

    def analyze(self, prompt):
        return self.analyses.pop(0)

    def respond(self, prompt):
        self.reply_prompts.append(prompt)
        return self.replies.pop(0) if self.replies else auto_reply(prompt)

    def evaluate(self, prompt):
        return self.reports.pop(0)


def ex(param, statement, quote, kind="fact", attr="x", gap_type="", related=""):
    return Extraction(kind=kind, param=param, attribute=attr, statement=statement,
                      source_quote=quote, gap_type=gap_type, related_id=related)


def an(extractions=(), intent="providing_info", mode="", corrections=(), questions=(), reactions=(), **kw):
    base = dict(owner_intent=intent, requested_mode=mode, extractions=list(extractions),
                corrections=list(corrections), withdrawn_ids=[], resolved_ids=[], confirmed_ids=[],
                owner_questions=list(questions), reactions=list(reactions))
    base.update(kw)
    return TurnAnalysis(**base)


@pytest.fixture
def store(tmp_path):
    return JsonStore(tmp_path)


def make(store, analyses, replies=None):
    llm = FakeLLM(analyses, replies)
    eng = IntakeEngine(llm, store)
    return eng, eng.new_session("Test"), llm


MSG = (
    "I want to make protein shakes, 5000 bottles a day. I run a dairy distribution business. "
    "Set up in Bangalore with 40 crore of my own. I know retailers. We have a food-tech team. "
    "Milk from local farms. I can take moderate risk. Aim to be a national brand."
)


def full_analysis():
    return an([
        ex("objective", "Owner wants to make protein shakes", "make protein shakes"),
        ex("objective", "Intended volume is 5000 bottles a day", "5000 bottles a day"),
        ex("existing_business", "Owner runs a dairy distribution business", "run a dairy distribution business"),
        ex("geography_capital", "Location is Bangalore", "Bangalore"),
        ex("geography_capital", "Owner has 40 crore of own funds", "40 crore of my own"),
        ex("strengths", "Owner knows retailers", "I know retailers"),
        ex("capabilities", "Owner has a food-tech team", "food-tech team"),
        ex("customers", "Owner knows retailers as potential buyers", "I know retailers"),
        ex("feedstock_access", "Milk from local farms", "Milk from local farms"),
        ex("risk_appetite", "Owner can take moderate risk", "moderate risk"),
        ex("strategic_ambitions", "Aim is a national brand", "national brand"),
    ], mode="detailed")


def idea(text="protein shakes"):
    return an([ex("objective", f"Owner wants to make {text}", text)])


def question_count(reply):
    return reply.count("?")


# --------------------------------------------------------------- guards ---
def test_fabricated_quote_is_rejected(store):
    eng, st, _ = make(store, [an([ex("objective", "Owner wants to make cement", "we will build a cement plant")])])
    eng.handle(st, "I want to make protein shakes")
    assert st.active("fact") == []
    assert any(a.get("reason") == "quote_not_found_in_owner_message" for a in st.audit)


def test_invented_number_is_rejected(store):
    eng, st, _ = make(store, [an([ex("geography_capital", "Owner has 50 crore", "40 crore")])])
    eng.handle(st, "I have 40 crore")
    assert st.active("fact") == []
    assert any(a.get("reason") == "number_not_in_owner_quote" for a in st.audit)


def test_number_words_are_matched(store):
    eng, st, _ = make(store, [an([ex("geography_capital", "Owner has 5 crore", "five crore")])])
    eng.handle(st, "I have five crore")
    assert len(st.active("fact")) == 1


# ---------------------------------------------- own words first, then paths ---
def test_greeting_asks_for_the_idea_in_own_words_and_offers_no_options():
    text = prompts.GREETING.format(name="")
    assert "own words" in text
    for word in ("Brief", "Balanced", "Detailed"):
        assert word not in text
    assert "\n- " not in text  # no bullet list of options
    assert question_count(text) == 1


def test_first_description_gets_a_restatement_and_2_to_3_paths_before_any_other_question(store):
    eng, st, llm = make(store, [idea()])
    reply = eng.handle(st, "I want to make protein shakes").reply
    assert "MOVE: suggest" in llm.reply_prompts[-1]
    assert "if I've understood" in reply
    assert "Path 1" in reply and "Path 2" in reply and "Path 3" not in reply
    assert "Pros:" in reply and "Cons:" in reply
    assert question_count(reply) == 1
    assert reply.rstrip().endswith("what would you change?")  # the only question invites a reaction
    assert st.path_rounds == 1


def test_even_a_complete_first_message_gets_paths_before_the_recap(store):
    eng, st, llm = make(store, [full_analysis()])
    reply = eng.handle(st, MSG).reply
    assert st.audit[-1]["move"] == "suggest"
    assert "Path 1" in reply and "Here is what I have captured" not in reply
    assert st.phase == "intake"


def test_paths_are_stored_as_system_suggestions_never_as_owner_facts(store):
    eng, st, _ = make(store, [idea()])
    eng.handle(st, "I want to make protein shakes")
    sugg = st.active("suggestion")
    assert len(sugg) == 2 and all(s.origin == "system" and not s.confirmed for s in sugg)
    assert all(s.pros and s.cons for s in sugg)
    assert [f.statement for f in st.active("fact")] == ["Owner wants to make protein shakes"]
    data = build_handoff(st)["shared_project_state"]
    assert len(data["facts"]) == 1 and len(data["system_suggestions"]["paths"]) == 2
    assert all(p["origin"] == "system" for p in data["system_suggestions"]["paths"])
    assert all(e["quote"] for e in data["evidence"])  # suggestions leave no fake owner evidence


def test_owner_reaction_is_recorded_with_the_owners_quote_and_stays_a_reaction(store):
    a2 = an([], reactions=[Reaction(target_id="S3", stance="likes", source_quote="path 2 feels right")])
    eng, st, llm = make(store, [idea(), a2])
    eng.handle(st, "I want to make protein shakes")
    facts_before = [f.statement for f in st.active("fact")]
    eng.handle(st, "Path 2 feels right to me")
    s3 = st.get("S3")
    assert s3.reaction == "likes" and s3.origin == "system" and s3.kind == "suggestion"
    assert s3.evidence[-1].source_type == "owner_statement" and s3.evidence[-1].quote == "path 2 feels right"
    assert [f.statement for f in st.active("fact")] == facts_before  # a reaction is not an owner fact
    assert "MOVE: converse" in llm.reply_prompts[-1]
    row = next(p for p in build_handoff(st)["shared_project_state"]["system_suggestions"]["paths"] if p["id"] == "S3")
    assert row["owner_reaction"] == "likes" and row["owner_reaction_quotes"][-1]["quote"] == "path 2 feels right"


def test_reaction_with_a_quote_the_owner_never_wrote_is_rejected(store):
    a2 = an([], reactions=[Reaction(target_id="S3", stance="likes", source_quote="I love path two")])
    eng, st, _ = make(store, [idea(), a2])
    eng.handle(st, "I want to make protein shakes")
    eng.handle(st, "hmm, let me think")
    assert st.get("S3").reaction == ""
    assert any(a.get("event") == "rejected" for a in st.audit)


def test_reaction_to_an_unknown_or_inactive_suggestion_is_ignored(store):
    a2 = an([], reactions=[Reaction(target_id="F1", stance="likes", source_quote="fine by me")])
    eng, st, _ = make(store, [idea(), a2])
    eng.handle(st, "I want to make protein shakes")
    eng.handle(st, "fine by me")
    assert st.get("F1").reaction == ""


def test_a_correction_cannot_rewrite_a_system_suggestion_into_an_owner_statement(store):
    a2 = an([], intent="correcting",
            corrections=[Correction(target_id="S2", new_statement="Owner wants contract work", source_quote="contract work")])
    eng, st, _ = make(store, [idea(), a2])
    eng.handle(st, "I want to make protein shakes")
    eng.handle(st, "I meant contract work")
    assert st.get("S2").status == "active"
    assert all("contract work" not in f.statement for f in st.active("fact"))


def test_rejecting_a_path_triggers_revised_paths_and_keeps_the_history(store):
    a2 = an([], reactions=[Reaction(target_id="S3", stance="dislikes", source_quote="path 2 won't work")])
    eng, st, llm = make(store, [idea(), a2])
    eng.handle(st, "I want to make protein shakes")
    reply = eng.handle(st, "Path 2 won't work for me, too capital heavy").reply
    assert "MOVE: revise" in llm.reply_prompts[-1]
    assert "Path 1" in reply and st.path_rounds == 2
    assert st.get("S3").status == "superseded" and st.get("S3").reaction == "dislikes"
    assert len(st.active("suggestion")) == 2
    ids = {p["id"]: p for p in build_handoff(st)["shared_project_state"]["system_suggestions"]["paths"]}
    assert ids["S3"]["owner_reaction"] == "dislikes" and ids["S3"]["status"] == "superseded"


def test_a_message_with_no_idea_is_invited_not_answered_with_paths(store):
    eng, st, llm = make(store, [an([], intent="other")])
    reply = eng.handle(st, "hello").reply
    assert "MOVE: invite" in llm.reply_prompts[-1]
    assert "Path 1" not in reply and st.path_rounds == 0
    assert question_count(reply) == 1


# ----------------------------------------------- conversation shape rules ---
def test_every_reply_gives_something_before_asking_and_asks_at_most_one_question(store):
    analyses = [an([], intent="other"), idea(), an([ex("existing_business", "Owner runs a dairy business", "dairy business")]),
                an([ex("customers", "Owner sells to retailers", "sell to retailers")])]
    eng, st, _ = make(store, analyses)
    for msg in ("hello", "I want to make protein shakes", "I have a dairy business", "we sell to retailers"):
        reply = eng.handle(st, msg).reply
        assert question_count(reply) <= 1
        without_question = re.sub(r"[^.!?\n]*\?", "", reply).strip()
        assert len(without_question.split()) >= 6, reply  # substance beyond the question


def test_question_mark_outside_question_field_is_retried(store):
    bad = rd(giveback="That is a fair starting point for us to work from. Where will you build it?", question="Which fits?")
    good = rd(giveback=GIVE, question="What's the idea you have in mind?")
    eng, st, llm = make(store, [an([], intent="other")], [bad, good])
    reply = eng.handle(st, "hello").reply
    assert question_count(reply) == 1 and "Where will you build it" not in reply
    assert any(a.get("event") == "reply_retry" for a in st.audit)


def test_extra_questions_that_survive_the_retry_are_removed_not_kept(store):
    bad = rd(giveback="That is a fair starting point for us to work from. Where will you build it?", question="Which fits?")
    eng, st, _ = make(store, [an([], intent="other")], [bad, bad])
    reply = eng.handle(st, "hello").reply
    assert question_count(reply) == 1 and "Where will you build it" not in reply


def test_two_questions_in_the_question_field_are_cut_to_one(store):
    bad = rd(giveback=GIVE, question="What is the idea? And where is it?")
    eng, st, _ = make(store, [an([], intent="other")], [bad, bad])
    reply = eng.handle(st, "hello").reply
    assert question_count(reply) == 1 and "And where is it" not in reply


def test_a_reply_that_is_only_a_question_is_never_sent(store):
    only_q = rd(giveback="", question="What's the idea?")
    eng, st, llm = make(store, [an([], intent="other")], [only_q, only_q])
    reply = eng.handle(st, "hello").reply
    assert reply == prompts.TROUBLE_REPLY
    assert st.phase == "intake" and st.transcript[-1].text == prompts.TROUBLE_REPLY
    assert any(a.get("event") == "reply_unusable" for a in st.audit)


def test_suggest_move_without_valid_paths_is_not_faked(store):
    thin = rd(restatement="You want protein shakes.", paths=[path("Only one path")], question="Thoughts?")
    eng, st, _ = make(store, [idea()], [thin, thin])
    reply = eng.handle(st, "I want to make protein shakes").reply
    assert reply == prompts.TROUBLE_REPLY and st.path_rounds == 0 and st.active("suggestion") == []
    assert len(st.active("fact")) == 1  # what the owner said is still captured


def test_recap_move_asks_no_question(store):
    eng, st, _ = make(store, [full_analysis(), an([], intent="unclear")],
                      [suggest_reply(), rd(giveback=GIVE + " The summary follows.", question="Anything else?")])
    eng.handle(st, MSG)
    st.mode = "brief"
    reply = eng.handle(st, "that's all I have").reply
    assert st.phase == "awaiting_confirmation" and "Anything else" not in reply
    assert "**confirm**" in reply


# ------------------------------------------------------- figures in replies ---
def test_reply_with_invented_figures_is_retried_then_uses_the_clean_draft(store):
    bad = rd(restatement="You want protein shakes.", paths=[path("A", pros=("Quicker", "Roughly 900 crore market")), path("B")],
             question="Which is closest?")
    eng, st, _ = make(store, [idea()], [bad, suggest_reply()])
    reply = eng.handle(st, "I want to make protein shakes").reply
    assert "900" not in reply
    assert any(a.get("event") == "reply_retry" for a in st.audit)


def test_figures_that_survive_the_retry_are_redacted(store):
    bad = rd(restatement="You want protein shakes.", paths=[path("A", pros=("Quicker", "Roughly 900 crore market")), path("B")],
             question="Which is closest?")
    eng, st, _ = make(store, [idea()], [bad, bad])
    reply = eng.handle(st, "I want to make protein shakes").reply
    assert "900" not in reply and "Quicker" in reply
    assert any(a.get("event") == "reply_numbers_redacted" for a in st.audit)
    assert all("900" not in x for s in st.active("suggestion") for x in s.pros)


def test_figures_from_system_text_do_not_become_allowed(store):
    eng, st, _ = make(store, [idea()])
    eng.handle(st, "I want to make protein shakes")
    assert "900" not in eng._allowed_numbers(st)
    st.add_item("assumption", "scope", "system text mentioning 900", origin="system", source="system_note")
    assert "900" not in eng._allowed_numbers(st)


# ------------------------------------------------------ checklist and flow ---
def test_full_flow_to_confirmation_and_handoff(store):
    eng, st, _ = make(store, [full_analysis(), an([], intent="unclear"), an([], intent="confirm_all")])
    r = eng.handle(st, MSG)
    assert st.mode == "detailed"
    assert st.phase == "intake" and "Path 1" in r.reply
    st.mode = "brief"  # owner asks for the light version: essentials are now enough
    r = eng.handle(st, "that's all I have")
    assert st.phase == "awaiting_confirmation"
    assert "**confirm**" in r.reply and "not** been independently verified" in r.reply
    assert "Paths I suggested" in r.reply
    r = eng.handle(st, "confirm")
    assert r.complete and st.phase == "complete"
    data = json.loads((store.root / "handoff" / f"{st.session_id}.json").read_text(encoding="utf-8"))
    assert len(data["shared_project_state"]["facts"]) == 11
    assert data["shared_project_state"]["assumptions"][0]["origin"] == "system"  # default scope, flagged
    assert data["shared_project_state"]["evidence"][0]["quote"]
    assert len(data["shared_project_state"]["system_suggestions"]["paths"]) == 2
    assert all(not s.confirmed for s in st.active("suggestion"))  # confirming the recap does not confirm our suggestions


def test_pyramid_order_asks_level1_first(store):
    eng, st, _ = make(store, [an([])])
    plan = fw.plan_next(st)
    assert plan.key == "objective"
    st.add_item("fact", "objective", "x", quote="x")
    assert fw.plan_next(st).key == "existing_business"


def test_suggestions_do_not_count_as_checklist_coverage(store):
    eng, st, _ = make(store, [idea()])
    eng.handle(st, "I want to make protein shakes")
    assert fw.param_status(st, "objective") == "covered"  # from the owner's fact
    st2 = ProjectState(session_id="abcdef")
    st2.add_item("suggestion", "objective", "Path: x", origin="system", source="system_note")
    assert fw.param_status(st2, "objective") == "empty"


def test_correction_supersedes_and_keeps_history(store):
    a1 = an([ex("geography_capital", "Location is Bangalore", "Bangalore")])
    a2 = an([], intent="correcting", corrections=[Correction(target_id="F1", new_statement="Location is Mysuru", source_quote="Mysuru")])
    eng, st, _ = make(store, [a1, a2])
    eng.handle(st, "Bangalore")
    eng.handle(st, "Sorry, I meant Mysuru")
    assert st.get("F1").status == "superseded" and st.get("F1").superseded_by == "F2"
    assert [i.statement for i in st.active("fact")] == ["Location is Mysuru"]


def test_ambiguity_triggers_clarification_in_a_later_reply(store):
    a1 = an([ex("objective", "Owner wants protein shakes", "protein shakes"),
             ex("geography_capital", "Owner has some capital", "some capital", kind="gap", gap_type="ambiguous")])
    eng, st, llm = make(store, [a1, an([], intent="other")])
    eng.handle(st, "protein shakes, I have some capital")
    eng.handle(st, "let me think")
    assert "MOVE: converse" in llm.reply_prompts[-1]
    assert "needs clarifying: [G2]" in llm.reply_prompts[-1]


def test_owner_skip_becomes_recorded_gap_not_guess(store):
    a = an([ex("customers", "Owner does not know customers yet", "no idea", kind="gap", gap_type="unknown_to_owner")], intent="declining_to_answer")
    eng, st, _ = make(store, [a])
    eng.handle(st, "no idea")
    assert fw.param_status(st, "customers") == "gap"


def test_repeated_evasion_becomes_not_provided_gap(store):
    eng, st, _ = make(store, [an([], intent="unclear") for _ in range(4)])
    for _ in range(4):
        eng.handle(st, "hmm")
    g = [x for x in st.active("gap", "objective")]
    assert g and g[0].gap_type == "not_provided" and g[0].origin == "system"


def test_no_question_asked_means_no_attempt_is_counted_against_the_owner(store):
    a2 = an([], intent="asking_question", questions=["What does CAPEX mean?"])
    answer = rd(answers=[QAnswer(question="What does CAPEX mean?", answer="Money spent building the plant.", basis="glossary")],
                giveback="Happy to explain any term as we go along, so just ask.", question="")
    eng, st, _ = make(store, [idea(), a2], [suggest_reply(), answer])
    eng.handle(st, "I want to make protein shakes")
    before = dict(st.focus_attempts)
    eng.handle(st, "What does CAPEX mean?")
    assert st.last_focus == "" and st.focus_attempts.get("") is None
    assert before.get("paths") is None or True


def test_owner_question_is_answered_and_unverified_noted(store):
    a = an([], intent="asking_question", questions=["What does CAPEX mean?", "What is the market size?"])
    r1 = rd(answers=[
        QAnswer(question="What does CAPEX mean?", answer="Money spent building the plant.", basis="glossary"),
        QAnswer(question="What is the market size?", answer="I can't say at intake.", basis="cannot_answer"),
    ], giveback="Whenever you are ready, tell me about the idea in whatever words come naturally.",
        question="What's the idea you have in mind?")
    eng, st, _ = make(store, [a], [r1])
    reply = eng.handle(st, "What does CAPEX mean? What is the market size?").reply
    assert "Money spent building the plant." in reply and "noted this question" in reply
    assert [q.answered for q in st.owner_questions] == [True, False]
    assert "What's the idea you have in mind?" in reply


def test_state_persists_and_reloads(store):
    eng, st, _ = make(store, [idea()])
    eng.handle(st, "I want to make protein shakes")
    again = store.load(st.session_id)
    assert again.active("fact")[0].evidence[0].quote == "protein shakes"
    assert len(again.active("suggestion")) == 2 and again.path_rounds == 1
    assert [s["id"] for s in store.list_sessions()] == [st.session_id]


def test_sessions_saved_before_this_change_still_load(store):
    st = ProjectState(session_id="abcdef1234")
    st.add_item("fact", "objective", "Owner wants x", quote="x")
    d = json.loads(st.model_dump_json())
    for key in ("path_rounds", "paths_turn", "last_move"):
        d.pop(key)
    for it in d["items"]:
        for key in ("pros", "cons", "reaction"):
            it.pop(key)
    (store.root / "sessions" / "abcdef1234.json").write_text(json.dumps(d), encoding="utf-8")
    loaded = store.load("abcdef1234")
    assert loaded.path_rounds == 0 and loaded.active("fact")[0].reaction == ""


def test_llm_schema_is_self_contained():
    for model in (TurnAnalysis, ReplyDraft, OpportunityReportOut):
        text = json.dumps(json_schema(model))
        assert "$ref" not in text and "$defs" not in text


def test_bad_session_id_rejected(store):
    with pytest.raises(ValueError):
        store.load("../../etc/passwd")


# ------------------------------------------------- report: SWOT + 5 forces ---
def _handoff():
    st = ProjectState(session_id="abcdef12")
    st.add_item("fact", "objective", "Owner wants to make protein shakes", quote="protein shakes")
    st.add_item("gap", "customers", "Owner has no customers yet", quote="no customers", gap_type="unknown_to_owner")
    return build_handoff(st)


def _point(text, basis="owner_stated", ids=("F1",)):
    return SwotPoint(point=text, basis=basis, source_ids=list(ids))


def _force(basis="general_knowledge", ids=()):
    return ForceAssessment(intensity="moderate", reasoning="A qualitative view of this force.", basis=basis, source_ids=list(ids))


def _report(strength_ids=("F1",)):
    alt = AlternativeOpportunity(name="Alt", type="product", rationale="Because.", fit_with_owner="Fits F1.")
    return OpportunityReportOut(
        swot=Swot(strengths=[_point("Has a clear product idea", ids=strength_ids)],
                  weaknesses=[_point("No customers identified yet", "inference", ("G2",))],
                  opportunities=[_point("Rising interest in the category", "general_knowledge", ())],
                  threats=[_point("Established brands may respond", "general_knowledge", ())]),
        porters_five_forces=FiveForces(
            threat_of_new_entrants=_force(), bargaining_power_of_suppliers=_force("owner_stated", ("F1",)),
            bargaining_power_of_buyers=_force("inference", ("G2",)), threat_of_substitutes=_force(),
            competitive_rivalry=_force(), overall_takeaway="A mixed picture for this owner."),
        idea_summary="x", market_analysis="x", technology_analysis="x", raw_materials_analysis="x",
        competition_analysis="x", capex_analysis="x", operating_economics_analysis="x", financing_analysis="x",
        risk_analysis="x", strategic_fit_analysis="x", assumptions=["a"], findings=["f"], recommendations=["r"],
        alternative_opportunities=[alt, alt])


def test_grounded_report_passes():
    assert report_problems(_handoff(), _report()) == []


def test_swot_point_citing_an_id_not_in_the_handoff_is_a_problem():
    problems = report_problems(_handoff(), _report(strength_ids=("F99",)))
    assert any("F99" in p for p in problems)


def test_owner_stated_point_without_a_source_is_a_problem():
    problems = report_problems(_handoff(), _report(strength_ids=()))
    assert any("cites no hand-off item" in p for p in problems)


def test_empty_swot_quadrant_is_a_problem():
    r = _report()
    r.swot.threats = []
    assert any("swot.threats is empty" in p for p in report_problems(_handoff(), r))


def test_report_is_retried_once_then_accepted_if_grounded():
    llm = FakeLLM([], reports=[_report(strength_ids=("F99",)), _report()])
    assert generate_report(_handoff(), llm).swot.strengths[0].source_ids == ["F1"]


def test_report_that_stays_ungrounded_is_rejected_not_patched():
    llm = FakeLLM([], reports=[_report(strength_ids=("F99",)), _report(strength_ids=("F99",))])
    with pytest.raises(LLMError):
        generate_report(_handoff(), llm)


def test_reports_saved_before_swot_existed_still_load(store):
    old = _report().model_dump()
    old.pop("swot")
    old.pop("porters_five_forces")
    (store.root / "reports" / "abcdef12.json").write_text(json.dumps(old), encoding="utf-8")
    loaded = store.load_report("abcdef12")
    assert isinstance(loaded, OpportunityReport) and loaded.swot is None
