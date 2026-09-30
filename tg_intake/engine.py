"""The conversation engine. Per owner message:
  1. analyze (LLM)  -> what did the owner say / how did they react to our suggestions?
  2. validate + apply to the Shared Project State (code; quote/number guards)
  3. decide the MOVE (code): invite | suggest | revise | converse | recap | await
     and plan_next (code) -> which open checklist item is worth working into the reply
  4. respond (LLM)  -> wording only; shape and numbers are checked by code
  5. persist to memory

Conversation rules enforced in code (not just requested in the prompt):
  - the owner describes the idea first; nothing is asked before something is given back
  - after the first description the reply restates the idea and offers 2-3 paths with pros and cons
  - every reply carries substantive content (a restatement, paths or a giveback), never a bare question
  - at most ONE question per reply, only in the dedicated field
  - paths are stored as SYSTEM suggestions; the owner's reactions are stored with their exact quote;
    nothing a suggestion says ever becomes an owner fact
"""
from dataclasses import dataclass, field

from . import framework as fw
from . import guards, prompts
from .llm import LLM, LLMError
from .schema import Evidence, OwnerQuestion, ProjectState, ReplyDraft, Turn, now
from .store import JsonStore

FALLBACK = "Sorry - I had trouble processing that. Could you say it again, perhaps in a slightly different way? Nothing has been changed."
GAP_TYPES = {"unknown_to_owner", "ambiguous", "conflict"}
MAX_PATH_ROUNDS = 4  # first suggestion + up to three revisions
MIN_GIVEBACK_WORDS = 6


@dataclass
class ApplyReport:
    changed: bool = False
    captured: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    new_questions: list = field(default_factory=list)
    reactions: list = field(default_factory=list)  # (suggestion item, stance)
    objective_changed: bool = False


@dataclass
class TurnResult:
    reply: str
    state: ProjectState
    complete: bool = False


class IntakeEngine:
    def __init__(self, llm: LLM, store: JsonStore, history_turns: int = 8):
        self.llm, self.store, self.history_turns = llm, store, history_turns

    # ------------------------------------------------------------ sessions --
    def new_session(self, owner_name: str = "", owner_email: str = "") -> ProjectState:
        import uuid

        st = ProjectState(session_id=uuid.uuid4().hex[:10], owner_name=owner_name.strip(), owner_email=owner_email)
        greeting = prompts.GREETING.format(name=f", {st.owner_name}" if st.owner_name else "")
        self._log(st, "assistant", greeting)
        st.last_bot_question = "What's the idea you have in mind?"
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
        move = self._decide_move(st, focus, report)
        if move == "recap":
            st.phase = "awaiting_confirmation"
        st.audit.append({"turn": st.turn, "event": "move", "move": move, "focus": focus.key})

        draft = self._draft_reply(st, text, report, focus, move)
        if draft is None:  # the model could not produce a valid reply; say so honestly, keep the state
            if move == "recap":
                st.phase = "intake"
            return self._finish(st, prompts.TROUBLE_REPLY)
        return self._finish(st, self._assemble(st, focus, move, report, draft))

    # ---------------------------------------------------------------- move --
    @staticmethod
    def _has_idea(st: ProjectState) -> bool:
        return bool(st.active("fact", "objective") or st.active("decision", "objective"))

    def _decide_move(self, st: ProjectState, focus: fw.Focus, report: ApplyReport) -> str:
        if focus.kind == "await_confirm":
            return "await"
        has_idea = self._has_idea(st)
        if not has_idea and fw.param_status(st, "objective") != "gap":
            return "invite"
        if has_idea and st.path_rounds == 0:
            return "suggest"  # always give something back before anything else, even if the owner told us everything
        if has_idea and st.path_rounds < MAX_PATH_ROUNDS and self._wants_revision(report):
            return "revise"
        if focus.kind == "recap":
            return "recap"
        return "converse"

    @staticmethod
    def _wants_revision(report: ApplyReport) -> bool:
        rejected = any(stance in ("dislikes", "wants_changed") for _, stance in report.reactions)
        return rejected or report.objective_changed

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

        # the owner's reaction to a system suggestion: recorded on the suggestion, with the owner's exact words.
        # It never turns the suggestion into an owner fact.
        for r in an.reactions:
            target = st.get(r.target_id)
            if not target or target.kind != "suggestion" or target.status != "active":
                continue
            if not guards.quote_in_message(r.source_quote, text):
                self._reject(st, rep, f"reaction to {r.target_id}", r.source_quote, "quote_not_found_in_owner_message")
                continue
            target.reaction = r.stance
            target.evidence.append(Evidence(source_type="owner_statement", turn=st.turn, quote=r.source_quote, at=now()))
            target.updated_at = now()
            rep.reactions.append((target, r.stance))
            rep.captured.append(f"[{target.id}] owner reaction to \"{target.attribute}\": {r.stance}")
            rep.changed = True

        if an.owner_intent in ("correcting", "providing_info"):
            for c in an.corrections:
                target = st.get(c.target_id)
                if not target or target.status != "active" or target.kind == "suggestion":
                    continue
                reason = self._check(c.source_quote, c.new_statement, text)
                if reason:
                    self._reject(st, rep, c.new_statement, c.source_quote, reason)
                    continue
                new = st.add_item(target.kind, target.param, c.new_statement, attribute=target.attribute,
                                  quote=c.source_quote, gap_type=target.gap_type, supersedes=target.id)
                target.status, target.superseded_by, target.updated_at = "superseded", new.id, now()
                rep.captured.append(f"[{new.id}] {new.statement} (replaces {target.id})")
                rep.changed = True
                if target.param == "objective":
                    rep.objective_changed = True
            for wid in an.withdrawn_ids:
                it = st.get(wid)
                if it and it.status == "active" and it.kind != "suggestion":
                    it.status, it.updated_at = "withdrawn", now()
                    rep.changed = True

        for rid in an.resolved_ids:
            it = st.get(rid)
            if it and it.kind == "gap" and it.status == "active":
                it.status, it.updated_at = "resolved", now()
                rep.changed = True
        for cid in an.confirmed_ids:
            it = st.get(cid)
            if it and it.status == "active" and it.kind != "suggestion":
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
        """Numbers the assistant may repeat: only ones the owner gave (never ones from system text)."""
        nums = set()
        for t in st.transcript:
            if t.role == "owner":
                nums |= guards.numbers(t.text)
        for i in st.items:
            if i.origin == "owner":
                nums |= guards.numbers(i.statement)
        return nums

    @staticmethod
    def _texts(d: ReplyDraft) -> list:
        """Every free-text field of a draft except `question`."""
        out = [a.answer for a in d.answers] + [d.restatement, d.giveback]
        for p in d.paths:
            out += [p.name, p.summary] + list(p.pros) + list(p.cons)
        return out

    def _draft_text(self, d: ReplyDraft) -> str:
        return " ".join(self._texts(d) + [d.question])

    def _problems(self, d: ReplyDraft, move: str, report: ApplyReport, allowed: set) -> list:
        problems = []
        bad = guards.unsupported_numbers(self._draft_text(d), allowed)
        if bad:
            problems.append(f"it contains figures the owner never gave: {sorted(bad)}. Use no figures that are not in <state> or <owner_message>")
        if any(guards.question_marks(t) for t in self._texts(d)):
            problems.append("a question mark appears outside the `question` field. Questions may appear only in `question`")
        if guards.question_marks(d.question) > 1:
            problems.append("`question` holds more than one question. Ask exactly one")
        problems += self._shape_problems(d, move, report)
        return problems

    @staticmethod
    def _shape_problems(d: ReplyDraft, move: str, report: ApplyReport) -> list:
        p = []
        if move in ("suggest", "revise"):
            if not d.restatement.strip():
                p.append("`restatement` is empty")
            if not 2 <= len(d.paths) <= 3:
                p.append("`paths` must hold 2 or 3 paths")
            for i, path in enumerate(d.paths, 1):
                if not (path.name.strip() and path.summary.strip()):
                    p.append(f"path {i} needs a name and a summary")
                if not [x for x in path.pros if x.strip()] or not [x for x in path.cons if x.strip()]:
                    p.append(f"path {i} needs at least one pro and one con")
            if not d.question.strip():
                p.append("`question` is empty; ask one open question inviting the owner's reaction")
        elif move in ("invite", "converse", "recap"):
            if len(d.giveback.split()) < MIN_GIVEBACK_WORDS:
                p.append("`giveback` is missing or too short; every reply must give something substantive, not only ask")
            if d.paths or d.restatement.strip():
                p.append("`paths` and `restatement` must be empty for this move")
            if move == "invite" and not d.question.strip():
                p.append("`question` is empty; invite the owner to describe the idea")
            if move == "converse" and not d.question.strip() and not report.new_questions:
                p.append("`question` is empty; ask exactly one question")
            if move == "recap" and d.question.strip():
                p.append("`question` must be empty for the recap")
        return p

    def _draft_reply(self, st, text, report, focus, move):
        """Ask the model, check the shape and figures in code, retry once with the exact violations.
        Returns None when no usable reply could be produced (the caller then says so honestly)."""
        prompt = prompts.reply_prompt(st, text, report, focus, move, self.history_turns)
        allowed = self._allowed_numbers(st)
        draft, problems = None, []
        for attempt in range(2):
            try:
                draft = self.llm.respond(prompt + (prompts.retry_note(problems) if problems else ""))
            except LLMError as exc:
                st.audit.append({"turn": st.turn, "event": "reply_failed", "error": str(exc)[:300]})
                return None
            problems = self._problems(draft, move, report, allowed)
            if not problems:
                break
            st.audit.append({"turn": st.turn, "event": "reply_retry" if attempt == 0 else "reply_still_invalid",
                             "problems": problems})

        # Deterministic clean-up. It only REMOVES text, it never writes any.
        bad = guards.unsupported_numbers(self._draft_text(draft), allowed)
        if bad:
            st.audit.append({"turn": st.turn, "event": "reply_numbers_redacted", "numbers": sorted(bad)})
            for a in draft.answers:
                a.answer = guards.redact_sentences_with(a.answer, bad)
            draft.restatement = guards.redact_sentences_with(draft.restatement, bad)
            draft.giveback = guards.redact_sentences_with(draft.giveback, bad)
            draft.question = guards.redact_sentences_with(draft.question, bad)
            for path in draft.paths:
                path.summary = guards.redact_sentences_with(path.summary, bad)
                path.pros = [x for x in path.pros if not (guards.numbers(x) & bad)]
                path.cons = [x for x in path.cons if not (guards.numbers(x) & bad)]
        for a in draft.answers:
            a.answer = guards.strip_question_sentences(a.answer)
        draft.restatement = guards.strip_question_sentences(draft.restatement)
        draft.giveback = guards.strip_question_sentences(draft.giveback)
        draft.question = guards.first_question(draft.question) if "?" in draft.question else draft.question.strip()
        if move in ("recap", "await"):
            draft.question = ""

        # Final gate on what cannot be repaired by removal: the reply must still carry content.
        hard = [x for x in self._shape_problems(draft, move, report) if "`question` is empty" not in x]
        if hard:
            st.audit.append({"turn": st.turn, "event": "reply_unusable", "problems": hard})
            return None
        return draft

    def _register_paths(self, st: ProjectState, paths) -> str:
        """Store the paths as SYSTEM suggestions (superseding the previous round) and render them for the owner."""
        for old in st.active("suggestion"):
            old.status, old.updated_at = "superseded", now()
        blocks = []
        for n, p in enumerate(paths, 1):
            pros = [x.strip() for x in p.pros if x.strip()]
            cons = [x.strip() for x in p.cons if x.strip()]
            st.add_item("suggestion", "objective", f"{p.name.strip()}: {p.summary.strip()}", attribute=p.name.strip(),
                        origin="system", source="system_note", pros=pros, cons=cons)
            blocks.append(
                f"**Path {n} - {p.name.strip()}**\n{p.summary.strip()}\n"
                f"- **Pros:** {'; '.join(pros)}\n- **Cons:** {'; '.join(cons)}"
            )
        st.path_rounds += 1
        st.paths_turn = st.turn
        return "\n\n".join(blocks)

    def _assemble(self, st, focus, move, report, draft) -> str:
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

        if move in ("suggest", "revise"):
            parts.append(draft.restatement.strip())
            parts.append(self._register_paths(st, draft.paths))
            parts.append(prompts.NOTE_PATHS)
        elif draft.giveback.strip():
            parts.append(draft.giveback.strip())

        if move == "recap":
            parts.append(fw.render_recap(st))
        elif move == "await":
            parts.append(prompts.CONFIRM_NUDGE)

        question = draft.question.strip()
        if question:
            parts.append(question)
        st.last_bot_question = question
        if move in ("suggest", "revise"):
            st.last_focus = "paths"
        elif move == "invite":
            st.last_focus = "objective"
        elif move == "converse":
            st.last_focus = focus.key if question else ""  # no question asked -> nothing to count against the owner
        else:
            st.last_focus = focus.key
        st.last_move = move
        return "\n\n".join(p for p in parts if p)

    # ----------------------------------------------------------- completion --
    def _complete(self, st: ProjectState) -> TurnResult:
        for it in st.active():
            if it.kind != "suggestion":  # the owner confirmed the recap, not our suggestions
                it.confirmed, it.confirmed_turn = True, st.turn
        st.phase = "complete"
        path = self.store.export_handoff(st)
        n = lambda k: len(st.active(k))  # noqa: E731
        msg = (
            "Thank you - your intake is confirmed and saved.\n\n"
            f"Recorded: {n('fact')} owner-stated facts, {n('assumption')} assumptions, {n('decision')} decisions, "
            f"{n('alternative')} alternatives and {n('gap')} open information gaps"
            f"{', plus the paths I suggested with your reactions to them' if n('suggestion') else ''}. "
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
