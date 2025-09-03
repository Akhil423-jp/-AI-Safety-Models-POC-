
# AI Safety Models POC (Lightweight)

This repository is a **self-contained, CPU-friendly** Proof of Concept that demonstrates:

- **Abuse Language Detection** (binary)
- **Escalation Pattern Recognition** (heuristic across recent messages)
- **Crisis Intervention Signal** (binary self-harm/ideation cues)
- **Age-Appropriate Content Filtering** (G / PG / PG-13 / R)

It uses **scikit-learn (TF-IDF + Logistic Regression)** and a simple **lexicon heuristic** for escalation.

## Quickstart

```bash
python -m pip install -r requirements.txt
# Train lightweight models
python -m src.train_abuse
python -m src.train_crisis
python -m src.train_age_filter
python -m src.train_escalation  # (evaluates heuristic on toy data)

# Evaluate
python -m src.evaluate

# Real-time demo (CLI)
python -m src.realtime_chat
```

## Project Structure

```
ai-safety-poc/
 ├─ data/                         # toy datasets (anonymized, tiny)
 ├─ models/                       # trained model artifacts (.joblib) and metrics
 ├─ src/                          # training, inference, realtime chat, evaluation
 ├─ report/                       # metrics + technical report (PDF & MD)
 ├─ README.md
 └─ requirements.txt
```

## Notes

- This is a **POC**: toy-size datasets and compact models for fast iteration.
- Swap in real datasets and re-run `src/train_*.py` scripts to improve accuracy.
- Ethical guardrails: bias and fairness are discussed in the Technical Report (`report/technical_report.pdf`).

