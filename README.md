# VoiceShield

## Anti-Spoofing for Voice-Based Authentication

VoiceShield is a mobile security application designed to help detect suspicious voice-based calls and potential voice impersonation attacks.

The application focuses on protecting users from vishing attacks where scammers may impersonate banks, organizations, or trusted individuals and ask for sensitive information such as OTPs or account details.

## Problem Statement

Voice-based scams and synthetic voice attacks are becoming increasingly difficult to identify. Attackers can use voice impersonation and social engineering techniques to convince users to reveal sensitive information.

VoiceShield aims to provide an additional layer of protection by analyzing caller voice patterns and conversation signals to identify potentially suspicious calls.

## Key Features

- Live voice detection interface
- Caller voice analysis concept
- HIGH RISK and LOW RISK detection results
- Suspicious OTP request detection
- Risk confidence percentage
- Detection history
- Save detection results to history
- Vibration alerts
- Protection ON/OFF setting
- Simple and user-friendly Android interface

## Example Detection

### HIGH RISK

> "I'm from your bank. Please tell me the OTP."

VoiceShield can display:

**⚠ HIGH RISK**

Possible voice impersonation / suspicious request detected.

**92% risk confidence**

### LOW RISK

For a normal conversation, the application can display:

**✓ LOW RISK**

No significant spoofing indicators detected.

## Technology Stack

- Kotlin
- Android Studio
- Jetpack Compose
- Material 3
- Android Vibration API
- Git & GitHub

## Current Status

The current version contains the Android frontend and a simulated detection flow.

The HIGH RISK and LOW RISK results are currently demonstration results. Actual real-time caller audio analysis and AI-based voice spoof detection will be integrated with the backend/model in a future version.

## Future Development

- Real-time microphone/audio processing
- AI-based synthetic voice detection
- Voice spoofing analysis
- Conversation risk analysis
- Backend API integration
- Real-time risk scoring
- VoIP and caller information analysis
- Persistent detection history
- Improved protection against voice-based social engineering attacks

## Target Users

- Banking and financial customers
- Senior citizens
- Enterprises and employees
- Banks and financial institutions
- Telecom and security providers
- General smartphone users

## Project Structure

```text
VoiceShield
│
├── app/
│   └── src/
│       └── main/
│           ├── java/
│           │   └── com/example/voicesheild/
│           │       └── MainActivity.kt
│           │
│           └── AndroidManifest.xml
│
└── README.md
