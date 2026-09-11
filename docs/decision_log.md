# Decision Log

This document records the major engineering and evaluation decisions made while building the Apple Support AI Agent. The goal is to make important trade-offs explicit, auditable, and reproducible.

---

## 1. Use a small, self-defined intent taxonomy

**Decision:** Define 10 support intents instead of attempting to reproduce a large existing taxonomy.

**Why:** The assignment asks for small, self-defined intents. A compact taxonomy makes classification easier to evaluate and makes the boundaries between intents explicit.

**Result:** The system uses:

- `software_os`
- `device_hardware`
- `connectivity`
- `account_auth`
- `apps_services`
- `setup_activation`
- `billing_purchase`
- `howto_feature`
- `privacy_security`
- `other`

---

## 2. Classify the customer's primary goal

**Decision:** Classification is based on the customer's primary support goal rather than individual keywords.

**Why:** The same keyword can appear in different support contexts. For example, a Wi-Fi problem caused by an OS update is different from a generic inability to connect.

**Result:** Boundary rules were documented for ambiguous cases such as OS updates, Apple ID issues during device setup, security incidents, and feature questions.

---

## 3. Include a majority-class baseline

**Decision:** Use a majority-class classifier as a simple reference baseline.

**Why:** A model's accuracy is difficult to interpret without a trivial reference point.

**Result:** The majority baseline achieved 48.67% accuracy on the 150-example golden set.

---

## 4. Include a rule-based baseline

**Decision:** Implement a simple keyword/rule-based classifier as a second baseline.

**Why:** This provides a stronger and more practical non-LLM comparison than majority class alone while remaining easy to inspect.

**Result:** The rule-based baseline achieved 46.67% accuracy.

---

## 5. Use Qwen 3.5 4B for classification and generation

**Decision:** Use the local `qwen3.5:4b` model through Ollama.

**Why:** The take-home permits open models, and a local model avoids requiring an external API key while keeping the project runnable in a fresh environment.

**Result:** The model can perform both structured intent classification and evidence-constrained response generation.

---

## 6. Use structured JSON for classification

**Decision:** Require the classifier to return structured classification output containing the predicted intent and confidence.

**Why:** Structured output makes the classifier easier to integrate with the agent and easier to evaluate programmatically.

**Result:** Classification results can be consumed deterministically by the escalation layer and evaluation scripts.

---

## 7. Use TF-IDF as the initial retrieval method

**Decision:** Use TF-IDF retrieval rather than introducing a more complex semantic retrieval stack.

**Why:** TF-IDF is lightweight, deterministic, locally runnable, and easy to inspect. This keeps the take-home focused on demonstrating the full support-agent pipeline rather than infrastructure complexity.

**Result:** The retriever indexes the historical customer/support corpus and returns scored evidence.

---

## 8. Treat historical support messages as a knowledge corpus

**Decision:** Do not assume the raw Twitter conversations represent clean customer → support resolution pairs.

**Why:** The source conversations are noisy and can contain unrelated issues and non-chronological relationships.

**Result:** Historical Apple Support messages are treated as a support knowledge corpus. Retrieval exposes historical customer context alongside the Apple Support response so the generator can distinguish the source of the resolution.

---

## 9. Weight support-response similarity more heavily

**Decision:** Use the final retrieval score:

`0.8 * support_response_similarity + 0.2 * customer_similarity`

**Why:** The final answer must be grounded in an actual historical Apple Support response. Therefore, similarity to the support response should carry more weight than similarity to the historical customer's wording.

**Result:** Retrieval ranking prioritizes support-resolution similarity while retaining a smaller contribution from historical customer similarity.

---

## 10. Constrain generation to retrieved evidence

**Decision:** The generation prompt explicitly restricts the model to the supplied historical evidence.

**Why:** Fluent model-generated troubleshooting can introduce unsupported claims. The goal is evidence-grounded drafting rather than general-purpose customer support generation.

**Result:** The generator is instructed not to invent troubleshooting steps, URLs, policies, guarantees, or actions. If evidence is insufficient, it can return an escalation marker instead of fabricating an answer.

---

## 11. Escalate high-risk security and billing cases

**Decision:** Treat `privacy_security` and `billing_purchase` as high-risk intents, with additional security phrase overrides.

**Why:** These cases can involve account compromise, unauthorized access, charges, refunds, or other situations where unsupported automated guidance is more costly.

**Result:** High-risk cases are escalated rather than automatically drafted as normal responses.

---

## 12. Prefer escalation when evidence is insufficient

**Decision:** The system should escalate when it cannot safely produce an evidence-grounded response.

**Why:** A high automation rate is less valuable than avoiding unsupported customer guidance.

**Result:** The escalation gate considers intent risk, classification confidence, evidence availability, and retrieval strength. The generator also has an explicit insufficient-evidence escape path.

---

## 13. Do not use the auto-handle rate as the primary quality metric

**Decision:** Treat the 92% auto-handle rate as a coverage/capacity metric rather than a quality metric.

**Why:** Passing the escalation gate does not prove that the retrieved evidence is relevant or that the final response is correct and helpful.

**Result:** Evaluation also measures classification performance, evidence grounding, response quality, escalation appropriateness, and critical failures.

---

## 14. Treat the LLM judge as diagnostic, not ground truth

**Decision:** Use the LLM-as-judge evaluation to identify failure patterns, but audit it with a manual second pass.

**Why:** An LLM judge can itself misunderstand evidence or conflate different quality dimensions.

**Result:** A 20-case manual review produced 60.0% raw agreement with the judge and Cohen's κ = 0.208. Because agreement was weak and disagreement analysis exposed judge errors, judge scores are treated as diagnostic signals rather than ground truth.

---

## 15. Use deterministic evaluation sampling

**Decision:** Construct the 150-example golden set and the judge subset deterministically using fixed seeds and explicit selection rules.

**Why:** Reproducible evaluation is necessary for comparing changes to the system without changing the test population.

**Result:** The evaluation artifacts can be regenerated and the same examples can be used for subsequent comparisons.

---

## Summary of the Main Trade-off

The project deliberately favors **simple, inspectable components and conservative handling** over maximum automation.

The evaluation showed that classification is reasonably strong, while retrieval and evidence quality are the larger bottlenecks. This led to the decision not to optimize the headline auto-handle rate further and instead prioritize evidence quality as the next engineering target.
