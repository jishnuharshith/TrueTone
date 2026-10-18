"""Local version 1 types. Keep field names aligned with the backend contract."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

LanguageHint = Literal["en", "hi", "auto"]
Language = Literal["en", "hi", "unknown"]
RiskLevel = Literal["insufficient_evidence", "low", "caution", "high"]
EvidenceLabel = Literal[
    "otp_request",
    "credential_request",
    "payment_request",
    "urgency",
    "threat",
    "impersonation",
    "remote_access_request",
    "suspicious_link",
]


@dataclass(frozen=True, slots=True)
class BehaviorWindow:
    schema_version: str
    session_id: str
    sequence: int
    language_hint: LanguageHint
    sample_rate_hz: int
    pcm_s16le: bytes
    window_start_ms: int
    window_end_ms: int

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("schema_version must be '1'")
        try:
            UUID(self.session_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("session_id must be a UUID string") from exc
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a nonnegative integer")
        if self.language_hint not in ("en", "hi", "auto"):
            raise ValueError("language_hint must be en, hi, or auto")
        if self.sample_rate_hz != 16000:
            raise ValueError("sample_rate_hz must be 16000")
        if not isinstance(self.pcm_s16le, bytes) or len(self.pcm_s16le) % 2:
            raise ValueError("pcm_s16le must be even-length bytes")
        if (
            type(self.window_start_ms) is not int
            or type(self.window_end_ms) is not int
            or self.window_start_ms < 0
            or self.window_end_ms <= self.window_start_ms
        ):
            raise ValueError("window timestamps must define a positive interval")


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    schema_version: str
    session_id: str
    sequence: int
    language: Language
    confidence: float
    risk_level: RiskLevel
    evidence_labels: list[EvidenceLabel]
    model_version: str
    inference_ms: int
