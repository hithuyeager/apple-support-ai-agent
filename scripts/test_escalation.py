from src.agent import SupportAgent

def print_result(result):
    print("\n=== SUPPORT AGENT ===")
    print(f"Message          : {result['message']}")
    print(f"Intent           : {result['intent']}")
    print(f"Confidence       : {result['confidence']:.2f}")
    print(f"Escalate         : {result['should_escalate']}")
    print(f"Risk level       : {result['risk_level']}")
    print(f"Escalation reason: {result['escalation_reason']}")
    print("\n=== DRAFT RESPONSE ===")
    print(result["draft_response"] or "[No autonomous draft — human handoff]")

def main():
    agent = SupportAgent()
    for message in [
        "My iPhone cannot activate. The activation server is unavailable.",
        "Someone got into my Apple ID and changed my password.",
        "I was charged twice for the same Apple purchase.",
        "How do I turn on dark mode on my iPhone?",
    ]:
        print_result(agent.handle(message))

if __name__ == "__main__":
    main()
