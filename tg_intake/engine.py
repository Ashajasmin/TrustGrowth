"""The conversation engine. Per owner message:
  1. analyze (LLM)  -> what did the owner say?
  2. validate + apply to the Shared Project State (code; quote/number guards)
  3. plan_next (code) -> what should we ask next? (pyramid, mode, gaps, clarifications)
  4. respond (LLM)  -> wording only; assembled + guarded by code
  5. persist to memory
"""
from dataclasses import dataclass, field

from . import framework as fw
from . import guards, prompts
from .llm import LLM, LLMError
from .schema import OwnerQuestion, ProjectState, Turn, now
from .store import JsonStore

FALLBACK = "Sorry - I had trouble processing that. Could you say it again, perhaps in a slightly different way? Nothing has been changed."
GAP_TYPES = {"unknown_to_owner", "ambiguous", "conflict"}


@dataclass
class ApplyReport:
    changed: bool = False
    captured: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    new_questions: list = field(default_factory=list)


@dataclass
class TurnResult:
    reply: str
    state: ProjectState
    complete: bool = False


class IntakeEngine:
    def __init__(self, llm: LLM, store: JsonStore, history_turns: int = 8):
        self.llm, self.store, self.history_turns = llm, store, history_turns

    # ------------------------------------------------------------ sessions --
    def new_session(self, owner_name: str = "") -> ProjectState:
        import uuid

        st = ProjectState(session_id=uuid.uuid4().hex[:10], owner_name=owner_name.strip())
        greeting = prompts.GREETING.format(name=f", {st.owner_name}" if st.owner_name else "")
        self._log(st, "assistant", greeting)
        st.last_bot_question = "What are you thinking of building or investing in?"
        st.last_focus = "objective"
        self.store.save(st)
        return st

    # -------------------------------------------------------------- a turn --
    def handle(self, st: ProjectState, text: str) -> TurnResult:
        text = (text or "").strip()
        if not text:
            return TurnResult("I did not catch anything - please type your answer.", st, st.phase == "complete")
        was_complete = st.phase == "complete"
        st.turn += 1
        if was_complete:
            st.phase = "intake"
        self._log(st, "owner", text)
        st.mode_chosen = True

        try:
            analysis = self.llm.analyze(prompts.analysis_prompt(st, text, self.history_turns))
        except LLMError as exc:
            st.audit.append({"turn": st.turn, "event": "analysis_failed", "error": str(exc)[:300]})
            if was_complete:
                st.phase = "complete"
            return self._finish(st, FALLBACK)

        report = self._apply(st, analysis, text)
        self._bump_attempts(st, analysis.owner_intent)

        if st.phase == "awaiting_confirmation":
            if analysis.owner_intent == "confirm_all" and not report.changed:
                return self._complete(st)
            if report.changed:
                st.phase = "intake"

        focus = fw.plan_next(st)
        if focus.kind == "recap":
            st.phase = "awaiting_confirmation"

        draft = self._draft_reply(st, text, report, focus)
        return self._finish(st, self._assemble(st, focus, report, draft))

    # ------------------------------------------------------------- applying --
    def _check(self, quote: str, statement: str, message: str) -> str:
        if not guards.quote_in_message(quote, message):
            return "quote_not_found_in_owner_message"
        if not guards.numbers_supported(statement, quote):
            return "number_not_in_owner_quote"
        return ""

    def _reject(self, st, rep, statement, quote, reason):
        rep.rejected.append({"statement": statement, "reason": reason})
        st.audit.append({"turn": st.turn, "event": "rejected", "reason": reason, "statement": statement, "quote": quote})

    def _apply(self, st: ProjectState, an, text: str) -> ApplyReport:
        rep = ApplyReport()
        if an.requested_mode and an.requested_mode != st.mode:
            st.audit.append({"turn": st.turn, "event": "mode_change", "from": st.mode, "to": an.requested_mode})
            st.mode = an.requested_mode

        for ex in an.extractions:
            reason = self._check(ex.source_quote, ex.statement, text)
            if reason:
                self._reject(st, rep, ex.statement, ex.source_quote, reason)
                continue
            if st.find_duplicate(ex.kind, ex.param, ex.statement):
                continue
            gap_type = (ex.gap_type if ex.gap_type in GAP_TYPES else "ambiguous") if ex.kind == "gap" else ""
            related = ex.related_id if st.get(ex.related_id) else ""
            item = st.add_item(ex.kind, ex.param, ex.statement, attribute=ex.attribute,
                               quote=ex.source_quote, gap_type=gap_type, related_id=related)
            rep.captured.append(f"[{item.id}] {item.statement}")
            rep.changed = True

        if an.owner_intent in ("correcting", "providing_info"):
            for c in an.corrections:
                target = st.get(c.target_id)
                reason = self._check(c.source_quote, c.new_statement, text)
                if not target or target.status != "active":
                    continue
                if reason:
                    self._reject(st, rep, c.new_statement, c.source_quote, reason)
                    continue
                new = st.add_item(target.kind, target.param, c.new_statement, attribute=target.attribute,
                                  quote=c.source_quote, gap_type=target.gap_type, supersedes=target.id)
                target.status, target.superseded_by, target.updated_at = "superseded", new.id, now()
                rep.captured.append(f"[{new.id}] {new.statement} (replaces {target.id})")
                rep.changed = True
            for wid in an.withdrawn_ids:
                it = st.get(wid)
                if it and it.status == "active":
                    it.status, it.updated_at = "withdrawn", now()
                    rep.changed = True

        for rid in an.resolved_ids:
            it = st.get(rid)
            if it and it.kind == "gap" and it.status == "active":
                it.status, it.updated_at = "resolved", now()
                rep.changed = True
        for cid in an.confirmed_ids:
            it = st.get(cid)
            if it and it.status == "active":
                it.confirmed, it.confirmed_turn = True, st.turn

        for q in an.owner_questions:
            if q.strip():
                oq = OwnerQuestion(turn=st.turn, question=q.strip())
                st.owner_questions.append(oq)
                rep.new_questions.append(oq)
        return rep

    def _bump_attempts(self, st: ProjectState, intent: str) -> None:
        if st.last_focus and intent in ("providing_info", "unclear", "declining_to_answer", "correcting"):
            st.focus_attempts[st.last_focus] = st.focus_attempts.get(st.last_focus, 0) + 1

    # ------------------------------------------------------------- replying --
    def _allowed_numbers(self, st: ProjectState) -> set:
        nums = set()
        for t in st.transcript:
            if t.role == "owner":
                nums |= guards.numbers(t.text)
        for i in st.items:
            nums |= guards.numbers(i.statement)
        return nums

    def _draft_reply(self, st, text, report, focus):
        from .schema import ReplyDraft

        prompt = prompts.reply_prompt(st, text, report, focus, self.history_turns)
        try:
            draft = self.llm.respond(prompt)
        except LLMError as exc:
            st.audit.append({"turn": st.turn, "event": "reply_failed", "error": str(exc)[:300]})
            return ReplyDraft(answers=[], acknowledgement="", question="")

        allowed = self._allowed_numbers(st)
        bad = guards.unsupported_numbers(self._draft_text(draft), allowed)
        if bad:
            st.audit.append({"turn": st.turn, "event": "reply_numbers_flagged", "numbers": sorted(bad)})
            try:
                draft = self.llm.respond(prompt + prompts.number_retry(bad))
            except LLMError:
                pass
            bad = guards.unsupported_numbers(self._draft_text(draft), allowed)
            if bad:  # last resort: remove the offending sentences
                st.audit.append({"turn": st.turn, "event": "reply_numbers_redacted", "numbers": sorted(bad)})
                for a in draft.answers:
                    a.answer = guards.redact_sentences_with(a.answer, bad)
                draft.acknowledgement = guards.redact_sentences_with(draft.acknowledgement, bad)
                draft.question = guards.redact_sentences_with(draft.question, bad)
        return draft

    @staticmethod
    def _draft_text(d) -> str:
        return " ".join([a.answer for a in d.answers] + [d.acknowledgement, d.question])

    def _assemble(self, st, focus, report, draft) -> str:
        parts = []
        answered = set()
        for idx, qa in enumerate(draft.answers[: len(report.new_questions)]):
            oq = report.new_questions[idx]
            oq.answer_basis = qa.basis
            oq.answered = qa.basis in ("intake_process", "glossary") and bool(qa.answer.strip())
            answered.add(idx)
            text = qa.answer.strip()
            if not text:
                oq.answer_basis, oq.answered = "cannot_answer", False
                text = "I can't answer that at the intake stage."
            if oq.answer_basis == "general_knowledge_unverified":
                text += prompts.NOTE_GENERAL
            elif not oq.answered:
                text += prompts.NOTE_CANNOT
            parts.append(text)
        for idx, oq in enumerate(report.new_questions):  # questions the model did not address
            if idx not in answered:
                oq.answer_basis = "cannot_answer"
                parts.append(f'I have noted your question ("{oq.question}") for the analysis stages.')

        if draft.acknowledgement.strip():
            parts.append(draft.acknowledgement.strip())

        if focus.kind == "recap":
            parts.append(fw.render_recap(st))
        elif focus.kind == "await_confirm":
            parts.append(prompts.CONFIRM_NUDGE)
        else:
            q = draft.question.strip() or fw.default_question(focus, st)
            parts.append(q)
            st.last_bot_question = q
        st.last_focus = focus.key
        return "\n\n".join(p for p in parts if p)

    # ----------------------------------------------------------- completion --
    def _complete(self, st: ProjectState) -> TurnResult:
        for it in st.active():
            it.confirmed, it.confirmed_turn = True, st.turn
        st.phase = "complete"
        path = self.store.export_handoff(st)
        n = lambda k: len(st.active(k))  # noqa: E731
        msg = (
            "Thank you - your intake is confirmed and saved.\n\n"
            f"Recorded: {n('fact')} owner-stated facts, {n('assumption')} assumptions, {n('decision')} decisions, "
            f"{n('alternative')} alternatives and {n('gap')} open information gaps. "
            "In the TrustGrowth architecture this becomes the shared project state that the analysis stages start from, "
            "and you review the results at each checkpoint.\n\n"
            f"Hand-off file: {path}\n\n"
            "If you remember something else or want to change anything, just tell me and I will reopen the intake."
        )
        return self._finish(st, msg)

    def _finish(self, st: ProjectState, reply: str) -> TurnResult:
        self._log(st, "assistant", reply)
        self.store.save(st)
        return TurnResult(reply, st, st.phase == "complete")

    @staticmethod
    def _log(st: ProjectState, role: str, text: str) -> None:
        st.transcript.append(Turn(n=len(st.transcript) + 1, role=role, text=text, at=now()))
