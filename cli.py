"""Terminal chat for the TG Opportunity Finder intake bot.

  python cli.py --owner "Mr. Goyal"     start a new intake
  python cli.py --list                  list saved sessions
  python cli.py --resume <session_id>   continue a saved session
Commands inside the chat: /state  /mode brief|balanced|detailed  /export  /help  /quit
"""
import argparse
import sys

from tg_intake import framework as fw
from tg_intake.config import get_settings
from tg_intake.engine import IntakeEngine
from tg_intake.llm import GeminiLLM, LLMError
from tg_intake.store import JsonStore

ICON = {"covered": "[x]", "gap": "[?]", "partial": "[~]", "empty": "[ ]"}


def show_state(st) -> None:
    print(f"\nSession {st.session_id} | mode: {st.mode} | phase: {st.phase}")
    for key, status in fw.coverage(st).items():
        print(f"  {ICON[status]} {fw.label_of(key)}  ({status})")
    for key in ("constraints", "scope"):
        n = len(st.active(param=key))
        print(f"  {'[x]' if n else '[ ]'} {fw.label_of(key)}")
    print("\nCaptured items:")
    for i in st.active():
        print(f"  [{i.id}] {i.kind}{':' + i.gap_type if i.gap_type else ''} | {i.param} | {i.statement}")
    print("  ([x] covered  [~] partial  [?] recorded as unknown  [ ] not yet asked)")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default="")
    ap.add_argument("--resume")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    store = JsonStore(settings.data_dir)
    if args.list:
        for s in store.list_sessions():
            print(f"{s['id']}  {s['owner'] or '-':20} {s['mode']:9} {s['phase']:22} {s['updated']}")
        return
    try:
        llm = GeminiLLM(settings)
    except LLMError as exc:
        sys.exit(str(exc))
    model_problems = llm.verify_models()
    if model_problems:
        sys.exit("Cannot start:\n  - " + "\n  - ".join(model_problems))
    engine = IntakeEngine(llm, store, settings.history_turns)

    if args.resume:
        st = store.load(args.resume)
        last = next((t.text for t in reversed(st.transcript) if t.role == "assistant"), "")
        print(f"[Resumed session {st.session_id}]\n\n{last}")
    else:
        st = engine.new_session(args.owner)
        print(f"[Session {st.session_id}]\n\n{st.transcript[-1].text}")

    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaved. Bye.")
            return
        if not text:
            continue
        if text == "/quit":
            print("Saved. Bye.")
            return
        if text == "/help":
            print(__doc__)
            continue
        if text == "/state":
            show_state(st)
            continue
        if text == "/export":
            print("Written to:", store.export_handoff(st))
            continue
        if text.startswith("/mode"):
            mode = text.split(maxsplit=1)[-1] if " " in text else ""
            if mode in fw.MODES:
                st.mode = mode
                store.save(st)
                print(f"Mode set to {mode}.")
            else:
                print("Usage: /mode brief|balanced|detailed")
            continue
        result = engine.handle(st, text)
        print(f"\nAssistant> {result.reply}")


if __name__ == "__main__":
    main()
