
from pathlib import Path
import joblib
from typing import Dict, Any
try:
    from .utils import normalize_text, moving_escalation, lexical_negativity_score
except Exception:
    from utils import normalize_text, moving_escalation, lexical_negativity_score

BASE = Path(__file__).resolve().parents[1]
_abuse = joblib.load(BASE / "models" / "abuse_model.joblib")
_crisis = joblib.load(BASE / "models" / "crisis_model.joblib")
_age = joblib.load(BASE / "models" / "age_model.joblib")

def infer_text(text: str, user_age: int = 16) -> Dict[str, Any]:
    t = normalize_text(text)
    abuse_pred = int(_abuse.predict([t])[0])
    crisis_pred = int(_crisis.predict([t])[0])
    age_class = str(_age.predict([t])[0])

    allowed = True
    if user_age <= 10 and age_class not in ("G",):
        allowed = False
    elif 11 <= user_age <= 13 and age_class not in ("G","PG"):
        allowed = False
    elif 14 <= user_age <= 17 and age_class not in ("G","PG","PG-13"):
        allowed = False

    return {
        "abuse": abuse_pred,
        "crisis": crisis_pred,
        "content_rating": age_class,
        "content_allowed_for_user": bool(allowed),
        "negativity_lexicon_hits": lexical_negativity_score(text),
    }

def infer_escalation(messages):
    return bool(moving_escalation(messages))
