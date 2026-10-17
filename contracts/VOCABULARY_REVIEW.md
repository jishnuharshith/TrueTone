# Shared Contract Vocabulary & Risk Meanings

> **STATUS: DRAFT FOR TEAM REVIEW (Backend, ML, Android)**
> **Last Updated**: 2026-10-17
> **Reviewers Required**: Backend Developer (Author), Shaun (ML Lead), Android Developer

---

## 1. Vocabulary Discrepancy Note

In initial planning discussions and the earlier audit summary, two slightly different candidate sets of evidence labels were referenced:
- **Audit Candidate Set**: `otp_request`, `urgency_threat`, `payment_transfer`, `app_install`, `impersonation` (5 labels).
- **Backend Master Prompt Working Set**: `otp_request`, `credential_request`, `payment_request`, `urgency`, `threat`, `impersonation`, `remote_access_request`, `suspicious_link` (8 labels).

### Decision for Draft v1:
Per project instructions, we have adopted the **8-label working vocabulary** from the backend master prompt as the baseline draft in [`contracts/models.py`](file:///c:/IBM-Z-Datathon/contracts/models.py). This granularity separates `urgency` from physical/legal `threat`, and distinguishes general `credential_request` from numeric `otp_request`.

**This vocabulary is NOT frozen.** It requires explicit sign-off from:
1. **Shaun (ML Lead)**: To verify that speech/text classification features and training/role-play evaluation fixtures can reliably support these 8 labels in both English and Hindi on LinuxONE CPU.
2. **Android Developer**: To verify that the UI card layouts and localized strings (English and Hindi) can present these labels cleanly during an active incoming call.

---

## 2. Risk Levels & Advisory Principles

All risk assessments are **strictly advisory**. The system assesses observed conversational cues in short audio windows; it never issues a definitive verdict or accusation against a caller.

| Risk Level | Draft Meaning | Suggested Android UI Copy (English) | Suggested Android UI Copy (Hindi) |
| :--- | :--- | :--- | :--- |
| `insufficient_evidence` | Silence, ambient noise, non-speech audio, low signal-to-noise ratio, or short speech where reliable assessment is not possible. The system defaults to this state rather than guessing. | "Not enough clear speech to assess yet." | "मूल्यांकन के लिए पर्याप्त स्पष्ट आवाज़ नहीं है।" |
| `low` | Conversational speech detected with **no indicators** of coercive, deceptive, or scam behavior in the available window. **CRITICAL**: This does **not** mean the caller is genuine or that the call is verified safe; it solely reflects the absence of detected risk indicators so far. | "No concerning behavior detected yet." | "अभी तक कोई संदेहास्पद व्यवहार नहीं मिला।" |
| `caution` | Observable conversational cues such as unverified claims of authority, mild urgency, or suspicious context detected. | "Be cautious. Verify caller identity independently." | "सावधान रहें। कॉलर की पहचान की पुष्टि करें।" |
| `high` | High-risk behavioral indicators detected (e.g., demanding OTP/PINs, threatening arrest, remote desktop install). | "Potential scam behavior. Do not share OTPs, PINs, or transfer money." | "संभावित धोखाधड़ी। OTP, PIN या पैसे साझा न करें।" |

---

## 3. Evidence Labels & Observable Speech Definitions

> **Key Rule**: Labels are defined strictly using **observable speech cues**. Acoustic and natural-language analysis cannot independently verify whether a caller is authorized or telling the truth; it can only detect the presence of specific claims, requests, or coercive patterns.

| Label | Observable Speech Definition | Typical Cues (English / Hindi) |
| :--- | :--- | :--- |
| `otp_request` | Caller explicitly asks for a one-time passcode, SMS verification number, or bank OTP. | "Tell me the 6-digit OTP you just received." / "जो 6 अंकों का OTP आया है वो बताएं।" |
| `credential_request` | Caller asks for sensitive authentication secrets (passwords, ATM PINs, CVV codes, net banking logins). | "Confirm your ATM PIN to unfreeze your account." / "अपना एटीएम पिन बताएं।" |
| `payment_request` | Caller requests or directs an immediate financial transaction (wire transfer, UPI, gift card code, crypto). | "Transfer Rs 5,000 immediately to clear the verification fee." / "तुरंत यूपीआई से भुगतान करें।" |
| `urgency` | Caller uses artificial time-compression language (claiming immediate account closure, limited time window). | "If you don't do this right now, your SIM card will be blocked within 15 minutes." / "10 मिनट में आपका खाता बंद हो जाएगा।" |
| `threat` | Caller uses coercive intimidation (threatening police arrest, CBI inquiry, legal action, power/utility cutoff). | "A non-bailable warrant has been issued against you by customs/police." / "आपके खिलाफ पुलिस कार्रवाई होगी।" |
| `impersonation` | Caller **claims to represent** an institution (bank, telecom provider, police/customs, government agency, or tech support). *Note: The system records the observable claim of representation; it does not label the caller as lying based on audio alone.* | "I am calling from RBI / Telecom Regulatory Authority verification cell." / "मैं बैंक हेड ऑफिस से बोल रहा हूँ।" |
| `remote_access_request` | Caller instructs or pressures the user to install remote desktop or screen-sharing software (AnyDesk, TeamViewer, QuickSupport). | "Install AnyDesk from Play Store so our technician can assist you." / "एनीडेस्क ऐप डाउनलोड करें।" |
| `suspicious_link` | Caller instructs the user to tap an unverified URL sent via SMS/messaging or navigate to an unofficial web portal. | "Click the link in the message I just sent to update your KYC." / "जो लिंक भेजा है उसपर तुरंत क्लिक करें।" |

---

## 4. Confidence & Insufficient Evidence Handling

- **Confidence Score (`0.0` to `1.0`)**: Represents the ML pattern matcher's certainty that the extracted acoustic and text features genuinely reflect the assigned behavioral labels and risk tier.
- **Handling Insufficient Evidence**:
  - When speech is absent, muted, below the signal-to-noise threshold, or too brief for pattern matching, the engine assigns `risk_level="insufficient_evidence"`, `evidence_labels=[]`, and confidence reflects the ambiguity.
  - The backend never guesses or applies priors to inflate risk during silent periods.
  - The Android client continues to display the advisory message ("Not enough clear speech to assess yet") rather than flipping rapidly between states.

---

## 5. Latency Target Reconciliation (<500 ms vs 800 ms Fallback)

The project documentation references two distinct latency thresholds:
1. **Target ML Window Inference Budget (`< 500 ms`)**:
   - **What it measures**: Pure internal compute latency of `analyze_window(request)` (ASR window transcription + rule/classifier feature extraction) on the LinuxONE s390x CPU.
   - **Purpose**: Ensures the backend can process 2.5-second audio windows with real-time headroom and push alerts to the mobile client within 1–2 seconds of caller speech.
2. **Backend Timeout Safeguard (`800 ms`)**:
   - **What it measures**: End-to-end backend timeout when invoking `analyze_window`.
   - **Purpose**: Prevents slow inference from stalling the media stream. If ML execution exceeds 800 ms, the backend drops the window, emits an advisory `call.error` or `insufficient_evidence`, and keeps the active Twilio call alive without audio degradation.
3. **Status**:
   - **Both values are strictly PROVISIONAL.**
   - Final latency thresholds will be calibrated and agreed upon once empirical benchmarks are run on the actual LinuxONE s390x VM with the chosen speech recognition model.

---

## 6. Privacy & Anti-Surveillance Safeguards

The data contracts strictly guarantee zero persistence and anti-surveillance compliance:
1. **No Transcripts**: Speech-to-text output is transiently processed in memory and discarded. No transcript field exists in `BehaviorResult` or `AppEvent`.
2. **No Audio Persistence**: Raw PCM audio bytes exist only within `BehaviorWindow` in the backend process during `analyze_window` execution and are immediately cleared. No media files are written to disk.
3. **No Caller Identity or PII**: Phone numbers, caller names, and quoted user speech are prohibited from all events.
4. **Session Ephemerality**: Session IDs are random UUIDs valid only for the lifespan of the single call leg and purged upon `call.ended`.
