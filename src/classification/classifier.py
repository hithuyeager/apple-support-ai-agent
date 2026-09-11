from dataclasses import dataclass
from typing import Protocol


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


@dataclass
class ClassificationResult:
    intent: str
    confidence: float


class Classifier(Protocol):
    def classify(self, text: str) -> ClassificationResult:
        ...