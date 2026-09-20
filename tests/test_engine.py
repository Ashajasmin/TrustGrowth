"""Engine tests with a scripted fake LLM (no network / API key needed).
Run:  python -m pytest -q"""
import json

import pytest

from tg_intake import framework as fw
from tg_intake.engine import IntakeEngine
from tg_intake.llm import json_schema
from tg_intake.schema import Correction, Extraction, QAnswer, ReplyDraft, TurnAnalysis
from tg_intake.store import JsonStore


class FakeLLM:
    def __init__(self, analyses, replies=None):
        self.analyses, self.replies, self.reply_prompts = list(analyses), list(replies or []), []

    def analyze(self, prompt):
        return self.analyses.pop(0)

    def respond(self, prompt):
        self.reply_prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ReplyDraft(answers=[], acknowledgement="", question="")


def ex(param, statement, quote, kind="fact", attr="x", gap_type="", related=""):
    return Extraction(kind=kind, param=param, attribute=attr, statement=statement,
                      source_quote=quote, gap_type=gap_type, related_id=related)


def an(extractions=(), intent="providing_info", mode="", corrections=(), questions=(), **kw):
    base = dict(owner_intent=intent, requested_mode=mode, extractions=list(extractions),
                corrections=list(corrections), withdrawn_ids=[], resolved_ids=[], confirmed_ids=[],
                owner_questions=list(questions))
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


def test_fabricated_quote_is_rejected(store):
    eng, st, _ = make(store, [an([ex("objective", "Owner wants to make cement", "we will build a cement plant")])])
    eng.handle(st, "I want to make protein shakes")
    assert st.active("fact") == []
    assert st.audit[-1]["reason"] == "quote_not_found_in_owner_message"


def test_invented_number_is_rejected(store):
    eng, st, _ = make(store, [an([ex("geography_capital", "Owner has 50 crore", "40 crore")])])
    eng.handle(st, "I have 40 crore")
    assert st.active("fact") == []
    assert st.audit[-1]["reason"] == "number_not_in_owner_quote"


def test_number_words_are_matched(store):
    eng, st, _ = make(store, [an([ex("geography_capital", "Owner has 5 crore", "five crore")])])
    eng.handle(st, "I have five crore")
    assert len(st.active("fact")) == 1


def test_full_flow_to_confirmation_and_handoff(store):
    eng, st, _ = make(store, [full_analysis(), an([], intent="confirm_all")])
    r = eng.handle(st, MSG)
    assert st.mode == "detailed"
    # detailed needs 2 facts per parameter, so it must keep asking, not finish
    assert st.phase == "intake" and "captured" not in r.reply.lower()
    eng.llm.analyses.insert(0, an([], intent="unclear"))
    st.mode = "brief"  # owner switches to brief: essentials are now enough
    r = eng.handle(st, "that's all I have")
    assert st.phase == "awaiting_confirmation"
    assert "**confirm**" in r.reply and "not** been independently verified" in r.reply
    r = eng.handle(st, "confirm")
    assert r.complete and st.phase == "complete"
    data = json.loads((store.root / "handoff" / f"{st.session_id}.json").read_text(encoding="utf-8"))
    assert len(data["shared_project_state"]["facts"]) == 11
    assert data["shared_project_state"]["assumptions"][0]["origin"] == "system"  # default scope, flagged
    assert data["shared_project_state"]["evidence"][0]["quote"]


def test_pyramid_order_asks_level1_first(store):
    eng, st, _ = make(store, [an([])])
    plan = fw.plan_next(st)
    assert plan.key == "objective"
    st.add_item("fact", "objective", "x", quote="x")
    assert fw.plan_next(st).key == "existing_business"


def test_correction_supersedes_and_keeps_history(store):
    a1 = an([ex("geography_capital", "Location is Bangalore", "Bangalore")])
    a2 = an([], intent="correcting", corrections=[Correction(target_id="F1", new_statement="Location is Mysuru", source_quote="Mysuru")])
    eng, st, _ = make(store, [a1, a2])
    eng.handle(st, "Bangalore")
    eng.handle(st, "Sorry, I meant Mysuru")
    assert st.get("F1").status == "superseded" and st.get("F1").superseded_by == "F2"
    assert [i.statement for i in st.active("fact")] == ["Location is Mysuru"]


def test_ambiguity_triggers_clarification_before_next_topic(store):
    a1 = an([ex("geography_capital", "Owner has some capital", "some capital", kind="gap", gap_type="ambiguous")])
    eng, st, llm = make(store, [a1])
    eng.handle(st, "I have some capital")
    assert "clarify item [G1]" in llm.reply_prompts[-1]


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


def test_owner_question_is_answered_and_unverified_noted(store):
    a = an([], intent="asking_question", questions=["What does CAPEX mean?", "What is the market size?"])
    r1 = ReplyDraft(answers=[
        QAnswer(question="What does CAPEX mean?", answer="Money spent building the plant.", basis="glossary"),
        QAnswer(question="What is the market size?", answer="I can't say at intake.", basis="cannot_answer"),
    ], acknowledgement="", question="")
    eng, st, _ = make(store, [a], [r1])
    reply = eng.handle(st, "What does CAPEX mean? What is the market size?").reply
    assert "Money spent building the plant." in reply and "noted this question" in reply
    assert [q.answered for q in st.owner_questions] == [True, False]
    assert "What are you thinking" not in reply and "product or business idea" in reply  # then continues intake


def test_reply_with_invented_figures_is_redacted(store):
    bad = ReplyDraft(answers=[], acknowledgement="The market is worth 900 crore.", question="Where would you set up?")
    eng, st, _ = make(store, [an([])], [bad, bad])
    reply = eng.handle(st, "hello").reply
    assert "900" not in reply and "Where would you set up?" in reply
    assert any(a["event"] == "reply_numbers_redacted" for a in st.audit)


def test_state_persists_and_reloads(store):
    eng, st, _ = make(store, [an([ex("objective", "Owner wants to make protein shakes", "protein shakes")])])
    eng.handle(st, "protein shakes")
    again = store.load(st.session_id)
    assert again.active("fact")[0].evidence[0].quote == "protein shakes"
    assert [s["id"] for s in store.list_sessions()] == [st.session_id]


def test_llm_schema_is_self_contained():
    for model in (TurnAnalysis, ReplyDraft):
        text = json.dumps(json_schema(model))
        assert "$ref" not in text and "$defs" not in text


def test_bad_session_id_rejected(store):
    with pytest.raises(ValueError):
        store.load("../../etc/passwd")
