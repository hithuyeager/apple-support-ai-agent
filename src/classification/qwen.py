import json

import ollama

from src.classification.classifier import ClassificationResult


INTENTS = [
    "software_os",
    "device_hardware",
    "connectivity",
    "account_auth",
    "apps_services",
    "setup_activation",
    "billing_purchase",
    "howto_feature",
    "privacy_security",
    "other",
]


SYSTEM_PROMPT = """
You are an intent classifier for an Apple customer-support system.

Your job is to classify the customer's PRIMARY problem or goal.

Choose exactly one intent from this list:

- software_os
- device_hardware
- connectivity
- account_auth
- apps_services
- setup_activation
- billing_purchase
- howto_feature
- privacy_security
- other

Important:
- Classify the customer's primary intent, not individual keywords.
- An iOS/update-related cause belongs to software_os even if the symptom involves
  battery, apps, performance, or connectivity.
- A physical device problem belongs to device_hardware.
- A problem connecting to WiFi/Bluetooth/network belongs to connectivity unless
  the customer is clearly blaming an OS/update.
- Apple ID/password/sign-in problems belong to account_auth unless they are part
  of setting up or activating a device.
- Problems during new-device setup, activation, restore, or activation lock belong
  to setup_activation.
- Purchases, charges, refunds, payments, or billing belong to billing_purchase.
- Questions about how to use a feature, feature availability, or requests for
  feature changes belong to howto_feature.
- Security, hacking, unauthorized access, or privacy concerns belong to
  privacy_security.
- App-specific/service-specific problems belong to apps_services.
- Use other when none of the categories reasonably fit.

Return ONLY valid JSON.
"""


class QwenClassifier:
    def __init__(self, model: str = "qwen3.5:4b"):
        self.model = model

    def classify(self, text: str) -> ClassificationResult:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                "role": "system",
                "content": SYSTEM_PROMPT,
                },
                {
                "role": "user",
                "content": text,
                },
        ],
        think=False,
        format={
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": INTENTS,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        },
        options={
            "temperature": 0,
        },
    )

        message = response["message"]

       

        if not message.get("content"):
            raise RuntimeError(
                f"Qwen returned empty content.\n"
                f"Full response: {response}"
            )

        data = json.loads(message["content"])

        return ClassificationResult(
            intent=data["intent"],
            confidence=float(data["confidence"]),
        )