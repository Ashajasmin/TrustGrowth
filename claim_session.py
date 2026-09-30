"""Give a saved conversation a verified owner.

Conversations created before e-mail verification existed (or with cli.py) have no owner e-mail, so the web app
cannot open them for anybody. Run this on the machine that holds the data folder to bind one to a mailbox; from
then on only a code sent to that mailbox opens it.

  python claim_session.py --list                     show every conversation and who owns it
  python claim_session.py <session_id> <email>       bind an unowned conversation to <email>
  python claim_session.py <session_id> <email> --force   re-assign one that already has an owner
"""
import argparse
import sys

from tg_intake.auth import AuthError, normalize_email
from tg_intake.config import get_settings
from tg_intake.store import JsonStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", nargs="?")
    ap.add_argument("email", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    store = JsonStore(get_settings().data_dir)
    if args.list:
        for s in store.list_sessions():
            st = store.load(s["id"])
            print(f"{st.session_id}  {st.owner_email or '(no owner)':32} {st.owner_name or '-':20} {st.phase:22} {st.updated_at}")
        return
    if not args.session_id or not args.email:
        ap.error("give a session id and an email, or use --list")
    try:
        email = normalize_email(args.email)
    except AuthError as exc:
        sys.exit(exc.message)
    try:
        st = store.load(args.session_id)
    except (FileNotFoundError, ValueError):
        sys.exit(f"No session {args.session_id}. Run with --list to see the ids.")
    if st.owner_email and st.owner_email != email and not args.force:
        sys.exit(f"{st.session_id} already belongs to {st.owner_email}. Use --force to re-assign it.")
    st.owner_email = email
    store.save(st)
    print(f"{st.session_id} now belongs to {email}.")


if __name__ == "__main__":
    main()
