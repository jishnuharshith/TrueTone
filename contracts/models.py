"""contracts/models.py

Pydantic v2 models for shared interfaces:
- BehaviorWindow and BehaviorResult (internal interface for analyze_window)
- AppEvent variants (WebSocket /v1/app/call-events push to Android)
- VoiceTokenResponse (POST /v1/voice/token response)

STATUS: DRAFT FOR TEAM REVIEW (Backend, ML, Android)
IMPORTANT:
- No raw audio, transcript, caller phone number, user name, or quoted speech is included.
- All media and provisional text remain in-memory only.
"""

from enum import Enum
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Vocabulary & Enums (DRAFT FOR TEAM REVIEW)
# ============================================================================

class RiskLevel(str, Enum):
    """Advisory risk level computed from caller behavior.

    STATUS: DRAFT FOR TEAM REVIEW (Advisory only; never a verdict on caller integrity)

    DRAFT MEANINGS FOR REVIEW:
    - insufficient_evidence: Silence, ambient noise, short speech, low audio quality,
      or uncertain recognition where speech cannot be reliably assessed. The system
      defaults to this state rather than guessing.
      Advisory note: 'Not enough clear speech to assess yet.'
    - low: Speech was processed and no concerning scam behavior was detected in the
      available window. IMPORTANT: This is an advisory baseline meaning 'no concerning
      indicators found'; it does NOT certify that the caller is genuine or that the call is safe.
      Advisory note: 'No concerning behavior detected yet.'
    - caution: Observable conversational cues such as unverified claims of authority,
      mild urgency, or suspicious context detected.
      Advisory note: 'Be cautious. Verify caller identity independently.'
    - high: High-risk behavioral indicators detected, such as demands for OTPs, PINs,
      passwords, coercive threats, or remote access installation requests.
      Advisory note: 'Potential scam behavior. Do not share OTPs, PINs, or transfer money.'
    """
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW = "low"
    CAUTION = "caution"
    HIGH = "high"


class WorkingEvidenceLabel(str, Enum):
    """Working vocabulary of detected behavioral scam indicators.

    STATUS: DRAFT FOR TEAM REVIEW
    (Backend working vocabulary aligned with master prompt; subject to sign-off with Shaun and Android).
    Labels capture observable speech cues only; audio analysis cannot determine caller veracity.

    DRAFT MEANINGS:
    - otp_request: Demanding one-time passcodes, SMS verification codes, or bank OTPs.
    - credential_request: Asking for passwords, PIN numbers, mother's maiden name, or login credentials.
    - payment_request: Requesting wire transfers, gift cards, crypto deposits, or UPI payment transfers.
    - urgency: Imposing false deadlines, claiming accounts will be blocked immediately without instant action.
    - threat: Threatening arrest, police action, legal penalties, utility cutoff, or financial seizure.
    - impersonation: Observable speech claiming to represent an institution (e.g. bank, telecom provider,
      police/law enforcement, government agency, or tech support). Does not claim to verify caller identity.
    - remote_access_request: Pressuring user to download AnyDesk, TeamViewer, QuickSupport, or remote management APKs.
    - suspicious_link: Coercing user to click an unverified SMS link or navigate to an unofficial web portal.
    """
    OTP_REQUEST = "otp_request"
    CREDENTIAL_REQUEST = "credential_request"
    PAYMENT_REQUEST = "payment_request"
    URGENCY = "urgency"
    THREAT = "threat"
    IMPERSONATION = "impersonation"
    REMOTE_ACCESS_REQUEST = "remote_access_request"
    SUSPICIOUS_LINK = "suspicious_link"


# Type alias allowing string flexibility while validating against working enum
EvidenceLabel = Union[WorkingEvidenceLabel, str]
DetectedLanguage = Literal["en", "hi", "unknown"]
LanguageHint = Literal["en", "hi", "auto"]


# ============================================================================
# ML Analyzer Interface: BehaviorWindow & BehaviorResult
# ============================================================================

class BehaviorWindow(BaseModel):
    """Input payload to analyze_window(request).

    Contains transient caller-only audio window. Processed in-memory; never persisted.
    Field names and types adhere strictly to the independent backend master prompt.
    """
    schema_version: Literal["1"] = "1"
    session_id: str = Field(
        ...,
        description="Opaque unique identifier for the active call session (e.g. UUIDv4)."
    )
    sequence: int = Field(
        ...,
        ge=0,
        description="Monotonically increasing sequence number for this call's audio windows."
    )
    language_hint: LanguageHint = Field(
        default="auto",
        description="Optional language hint for ASR (en, hi, or auto)."
    )
    sample_rate_hz: int = Field(
        default=16000,
        description="Sampling rate in Hz (standard 16000 Hz)."
    )
    pcm_s16le: bytes = Field(
        ...,
        description="Mono signed 16-bit linear PCM audio bytes of the external caller only."
    )
    window_start_ms: int = Field(
        ...,
        ge=0,
        description="Start offset of this audio window in milliseconds from call start."
    )
    window_end_ms: int = Field(
        ...,
        ge=0,
        description="End offset of this audio window in milliseconds from call start."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BehaviorResult(BaseModel):
    """Output result returned by analyze_window(request).

    Zero transcript, audio, quote, phone number, or caller identity is present.
    """
    schema_version: Literal["1"] = "1"
    session_id: str = Field(
        ...,
        description="Matches the input BehaviorWindow session_id."
    )
    sequence: int = Field(
        ...,
        ge=0,
        description="Matches the input BehaviorWindow sequence number."
    )
    language: DetectedLanguage = Field(
        ...,
        description="Detected spoken language of the caller window (en, hi, or unknown)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Internal model confidence score (0.0 to 1.0) for the detected behavior. "
            "Reflects the pattern matcher's certainty given the audio quality and extracted cues. "
            "When evidence is ambiguous or speech is unclear, risk_level defaults to "
            "insufficient_evidence."
        )
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Computed risk level (insufficient_evidence, low, caution, high)."
    )
    evidence_labels: List[EvidenceLabel] = Field(
        default_factory=list,
        description="Observed scam behavior labels from the working vocabulary."
    )
    model_version: str = Field(
        ...,
        description="Identifier of model/rule-engine version used for inference."
    )
    inference_ms: float = Field(
        ...,
        ge=0.0,
        description="Execution latency of analyze_window in milliseconds."
    )


# ============================================================================
# Mobile App Event Models (WebSocket WS /v1/app/call-events)
# ============================================================================

class AppEventBase(BaseModel):
    """Common base for all live call events dispatched to the Android app."""
    schema_version: Literal["1"] = "1"
    session_id: str = Field(
        ...,
        description="Opaque UUID identifying the active call session."
    )
    sequence: int = Field(
        ...,
        ge=0,
        description="Monotonically increasing sequence number for client event ordering."
    )


class CallRingingEvent(AppEventBase):
    """Dispatched when Twilio Voice webhook notifies that the incoming call is ringing."""
    type: Literal["call.ringing"] = "call.ringing"
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds when ringing started."
    )


class CallConnectedEvent(AppEventBase):
    """Dispatched when the call media stream establishes and the call is connected."""
    type: Literal["call.connected"] = "call.connected"
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds when call connected."
    )


class CallRiskUpdateEvent(AppEventBase):
    """Dispatched when a new scam-behavior assessment is computed for the active call."""
    type: Literal["call.risk.update"] = "call.risk.update"
    language: DetectedLanguage = Field(
        ...,
        description="Detected language of caller's latest speech (en, hi, or unknown)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the assessment between 0.0 and 1.0."
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Advisory risk level for the UI."
    )
    evidence_labels: List[EvidenceLabel] = Field(
        default_factory=list,
        description="Active behavioral risk labels (e.g. ['otp_request', 'urgency'])."
    )
    updated_at_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds when this assessment was generated."
    )


class CallEndedEvent(AppEventBase):
    """Dispatched when the call terminates (caller hangup, user hangup, or rejection)."""
    type: Literal["call.ended"] = "call.ended"
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds when call ended."
    )
    reason: Optional[Literal["caller_hangup", "user_hangup", "rejected", "error"]] = Field(
        default=None,
        description="Optional high-level termination reason."
    )


class CallErrorEvent(AppEventBase):
    """Dispatched when a non-fatal or fatal error occurs in processing."""
    type: Literal["call.error"] = "call.error"
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds of the error event."
    )
    error_code: str = Field(
        ...,
        description="Standardized error code (e.g., 'STREAM_TIMEOUT', 'ML_UNAVAILABLE')."
    )
    message: str = Field(
        ...,
        description="Safe, non-sensitive advisory explanation."
    )


# Discriminated Union for polymorphic deserialization on client & server
AppEvent = Annotated[
    Union[
        CallRingingEvent,
        CallConnectedEvent,
        CallRiskUpdateEvent,
        CallEndedEvent,
        CallErrorEvent,
    ],
    Field(discriminator="type")
]


# ============================================================================
# Twilio Voice Token Response (POST /v1/voice/token)
# ============================================================================

class VoiceTokenResponse(BaseModel):
    """Response returned to Android app to authorize Twilio Voice SDK registration."""
    schema_version: Literal["1"] = "1"
    token: str = Field(
        ...,
        description="Short-lived signed Twilio Voice Access Token (JWT)."
    )
    identity: str = Field(
        ...,
        description="Registered client identity (e.g., 'client:android_user')."
    )
    expires_in_seconds: int = Field(
        default=3600,
        gt=0,
        description="Remaining validity duration of the token in seconds."
    )
    issued_at_ms: int = Field(
        ...,
        ge=0,
        description="Unix timestamp in milliseconds when token was generated."
    )
