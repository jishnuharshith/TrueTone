"""Small transparent rules for operator-written text; not a trained model."""

from dataclasses import dataclass
import re

from .models import EvidenceLabel, Language, LanguageHint, RiskLevel


@dataclass(frozen=True, slots=True)
class TextAssessment:
    language: Language
    confidence: float
    risk_level: RiskLevel
    evidence_labels: list[EvidenceLabel]
    model_version: str = "text-prototype-v1"


_ASK = r"(?:\b(?:tell|share|give|send|read|provide|batao|bataiye|bhejo)\b|बताओ|बताइए|भेजो|दो|दीजिए)"
_OTP = r"(?:\b(?:otp|one[ -]?time(?: password| code)?|verification code)\b|ओटीपी|वेरिफिकेशन कोड)"
_CREDENTIAL = r"(?:\b(?:pin|password|passcode|cvv)\b|पासवर्ड|पिन|सीवीवी)"
# Whisper can render spoken Hindi in Urdu script even when language is forced
# to Hindi. Recognize the OTP/tell forms observed in our consented ASR check.
_ASK = rf"(?:{_ASK}|ب[تط]ا(?:ؤ|و|ل)|بتائیے)"
_OTP = rf"(?:{_OTP}|(?:او|ا)\s*ٹی\s*پی|एटिपी)"
_NEGATED_URDU_ADVICE = re.compile(r"(?<!\w)(?:مت|نہیں|نہ)(?!\w)")
_NEGATED_ADVICE = re.compile(
    r"\b(?:never|don'?t|do not|not|shouldn'?t|mat|nahi|nahin)\b|(?:मत|नहीं|ना दें|न बताएं)",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"[.!?;,\n।]+|\b(?:but|however|lekin)\b|लेकिन|परन्तु", re.IGNORECASE)


def _requested(text: str, target: str) -> bool:
    """Require a request verb near a sensitive target in one clause."""
    return bool(
        re.search(rf"{_ASK}.{{0,50}}{target}|{target}.{{0,50}}{_ASK}", text, re.IGNORECASE)
    )


def _language(text: str, hint: LanguageHint) -> Language:
    if hint in ("en", "hi"):
        return hint
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"[\u0600-\u06ff]", text):
        return "unknown"
    if re.search(r"\b(?:aap|apna|mera|mujhe|batao|bhejo|jaldi|abhi|paisa|paise|karo|nahi|kholo)\b", text):
        return "hi"
    return "en"


def classify_text(text: str, language_hint: LanguageHint = "auto") -> TextAssessment:
    """Assess a scripted sentence in memory; never log or return the sentence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if language_hint not in ("en", "hi", "auto"):
        raise ValueError("language_hint must be en, hi, or auto")
    normalized = " ".join(text.casefold().split())
    if len(normalized) < 8 or len(normalized.split()) < 2:
        return TextAssessment("unknown", 0.0, "insufficient_evidence", [])

    found: set[EvidenceLabel] = set()
    for clause in _CLAUSE_BOUNDARY.split(normalized):
        clause = clause.strip()
        if not clause or _NEGATED_ADVICE.search(clause) or _NEGATED_URDU_ADVICE.search(clause):
            continue
        if _requested(clause, _OTP):
            found.add("otp_request")
        if _requested(clause, _CREDENTIAL):
            found.add("credential_request")
        if re.search(
            r"\b(?:transfer|pay|send|wire|bhejo)\b.{0,45}\b(?:money|cash|funds|rupees|rs|paisa|paise|fee|charge)\b"
            r"|\b(?:money|cash|funds|rupees|paisa|paise)\b.{0,45}\b(?:transfer|pay|send|bhejo)\b"
            r"|(?:पैसे|रुपये|भुगतान).{0,45}(?:भेजो|भेजें|जमा|करो|दीजिए)",
            clause,
        ):
            found.add("payment_request")
        if re.search(r"\b(?:immediately|right now|urgent|hurry|jaldi|abhi|before noon|before midnight)\b|(?:तुरंत|अभी|जल्दी)", clause):
            found.add("urgency")
        if re.search(r"\b(?:arrest|suspend|block your account|account will be blocked|police case|legal action)\b|(?:गिरफ्तार|खाता बंद|पुलिस केस)", clause):
            found.add("threat")
        if re.search(
            r"\b(?:i am|i'm|calling from)\b.{0,35}\b(?:bank|police|government|support|customer care)\b"
            r"|\b(?:bank se bol|police se bol|sarkar se bol)\b"
            r"|(?:बैंक|पुलिस|सरकार).{0,30}(?:बोल|अधिकारी|से)",
            clause,
        ):
            found.add("impersonation")
        if re.search(
            r"\b(?:install|download|screen share|remote access)\b.{0,35}\b(?:app|application|anydesk|teamviewer|software)\b"
            r"|\b(?:app|application|anydesk|teamviewer)\b.{0,30}\b(?:install|download|karo)\b"
            r"|(?:ऐप|एप).{0,30}(?:इंस्टॉल|डाउनलोड|करो|कीजिए)",
            clause,
        ):
            found.add("remote_access_request")
        if re.search(
            r"\b(?:click|open|tap|kholo)\b.{0,25}\b(?:link|url|website)\b"
            r"|\b(?:link|url|website)\b.{0,25}\b(?:click|open|tap|kholo)\b"
            r"|(?:लिंक).{0,25}(?:खोलो|खोलें|क्लिक|दबाओ)",
            clause,
        ):
            found.add("suspicious_link")

    order: tuple[EvidenceLabel, ...] = (
        "otp_request", "credential_request", "payment_request", "urgency",
        "threat", "impersonation", "remote_access_request", "suspicious_link",
    )
    labels = [label for label in order if label in found]

    if any(label in labels for label in ("otp_request", "credential_request", "payment_request", "remote_access_request")):
        risk: RiskLevel = "high"
        confidence = 0.75
    elif len(labels) >= 2:
        risk = "high"
        confidence = 0.70
    elif labels:
        risk = "caution"
        confidence = 0.60
    else:
        risk = "low"
        confidence = 0.55
    return TextAssessment(_language(normalized, language_hint), confidence, risk, labels)
