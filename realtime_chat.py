
try:
    from .inference import infer_text, infer_escalation
except Exception:
    from inference import infer_text, infer_escalation

def main():
    print("AI Safety POC – Real-time Chat (toy demo). Type '/quit' to exit.")
    user_age = 15
    history = []
    while True:
        try:
            msg = input("You: ").strip()
        except EOFError:
            break
        if msg == "/quit":
            break
        history.append(msg)
        if len(history) > 3:
            history = history[-3:]
        res = infer_text(msg, user_age=user_age)
        print(f"  -> Abuse: {res['abuse']} | Crisis: {res['crisis']} | Rating: {res['content_rating']} | Allowed: {res['content_allowed_for_user']} | NegHits: {res['negativity_lexicon_hits']}")
        if len(history) == 3 and infer_escalation(history):
            print("  !! Escalation detected across last 3 messages -> consider human handoff")

if __name__ == '__main__':
    main()
