
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

try:
    from .utils import moving_escalation
except Exception:
    from utils import moving_escalation

def main():
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(base / "data" / "escalation_conversations_toy.csv")
    y_true = df["label"].tolist()
    y_pred = [1 if moving_escalation([m1,m2,m3]) else 0 for m1,m2,m3 in zip(df["m1"], df["m2"], df["m3"])]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    (base / "models" / "escalation_metrics.json").write_text(pd.Series(report).to_json())

if __name__ == "__main__":
    main()
