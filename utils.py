
import re
from typing import List

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s'!?.,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

NEGATIVE_LEXICON = set([
    "hate","idiot","loser","moron","worthless","stupid","punch","shut","trash",
    "hurt","kill","end","hopeless","nothing","die","suicide","self-harm","end it"
])

def lexical_negativity_score(text: str) -> int:
    t = normalize_text(text)
    return sum(1 for w in t.split() if w in NEGATIVE_LEXICON)

def moving_escalation(messages: List[str]) -> bool:
    # Simple heuristic: if negativity strictly increases across messages
    scores = [lexical_negativity_score(m) for m in messages]
    return scores[0] <= scores[1] <= scores[2] and scores[-1] >= 2
