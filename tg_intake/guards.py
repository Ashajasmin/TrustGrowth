"""Deterministic anti-hallucination checks. The LLM is never trusted on its own:
- anything stored must be traceable to an exact quote from the owner, and
- numbers may only come from the owner's own words."""
import re

_WORD_NUMS = {
    w: n
    for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_WORD_NUMS.update(
    {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
SMALL = {str(i) for i in range(0, 11)}  # harmless in replies ("2 questions", "step 3")


def normalize(text: str) -> str:
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    )
    return " ".join(text.lower().split())


def quote_in_message(quote: str, message: str) -> bool:
    q = normalize(quote)
    return bool(q) and q in normalize(message)


def numbers(text: str) -> set[str]:
    out = {tok.replace(",", "").rstrip(".") for tok in _NUM_RE.findall(text)}
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _WORD_NUMS:
            out.add(str(_WORD_NUMS[w]))
    return out


def numbers_supported(statement: str, quote: str) -> bool:
    """Every number in the statement must also be in the owner's quote."""
    return numbers(statement) <= numbers(quote)


def unsupported_numbers(reply_text: str, allowed: set[str]) -> set[str]:
    """Numbers in an assistant reply that the owner never gave us."""
    return numbers(reply_text) - allowed - SMALL


def redact_sentences_with(text: str, bad: set[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = [s for s in sentences if not (numbers(s) & bad)]
    return " ".join(keep).strip()


# ---------------------------------------------------------------- shape ---
# Conversation-shape rules: at most one question per reply, and it may live only
# in the dedicated `question` field.
def question_marks(text: str) -> int:
    return (text or "").count("?")


def strip_question_sentences(text: str) -> str:
    """Remove every sentence that ends in a question mark (only removes, never adds)."""
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    return " ".join(s for s in sentences if not s.rstrip().endswith("?")).strip()


def first_question(text: str) -> str:
    """Keep only the first question sentence of a text ("" if it has none)."""
    for s in re.split(r"(?<=[.!?])\s+", text or ""):
        if s.rstrip().endswith("?"):
            return s.strip()
    return (text or "").strip() if "?" not in (text or "") else ""
