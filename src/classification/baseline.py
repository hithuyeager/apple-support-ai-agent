from .classifier import ClassificationResult


class RuleBasedClassifier:
    def classify(self, text: str) -> ClassificationResult:
        text = text.lower()

        # Software / OS
        if any(term in text for term in [
            "ios",
            "update",
            "software",
            "after updating",
            "after the update",
            "ios 11",
        ]):
            return ClassificationResult(
                intent="software_os",
                confidence=0.70,
            )

        # Setup / activation
        if any(term in text for term in [
            "activation",
            "activate",
            "setting up",
            "setup",
            "restore my iphone",
            "new iphone",
        ]):
            return ClassificationResult(
                intent="setup_activation",
                confidence=0.70,
            )

        # Connectivity
        if any(term in text for term in [
            "wifi",
            "wi-fi",
            "bluetooth",
            "network",
            "can't connect",
            "cannot connect",
        ]):
            return ClassificationResult(
                intent="connectivity",
                confidence=0.70,
            )

        # Account / authentication
        if any(term in text for term in [
            "apple id",
            "password",
            "sign in",
            "signin",
            "login",
            "log in",
        ]):
            return ClassificationResult(
                intent="account_auth",
                confidence=0.70,
            )

        # Billing / purchases
        if any(term in text for term in [
            "charged",
            "charge",
            "payment",
            "refund",
            "purchase",
            "billing",
            "credit card",
            "debit card",
        ]):
            return ClassificationResult(
                intent="billing_purchase",
                confidence=0.70,
            )

        # Apps / services
        if any(term in text for term in [
            "app store",
            "appstore",
            "app",
            "itunes",
            "icloud",
            "podcast",
        ]):
            return ClassificationResult(
                intent="apps_services",
                confidence=0.60,
            )

        # Hardware
        if any(term in text for term in [
            "screen",
            "display",
            "camera",
            "speaker",
            "microphone",
            "headphone",
            "button",
            "broken",
            "cracked",
        ]):
            return ClassificationResult(
                intent="device_hardware",
                confidence=0.60,
            )

        # How-to / feature requests
        if any(term in text for term in [
            "how do i",
            "how can i",
            "can i",
            "please add",
            "please make",
            "is there a way",
            "where can i",
        ]):
            return ClassificationResult(
                intent="howto_feature",
                confidence=0.60,
            )

        # Privacy / security
        if any(term in text for term in [
            "hacked",
            "stolen",
            "privacy",
            "security",
            "unauthorized",
            "someone accessed",
        ]):
            return ClassificationResult(
                intent="privacy_security",
                confidence=0.70,
            )

        return ClassificationResult(
            intent="other",
            confidence=0.30,
        )