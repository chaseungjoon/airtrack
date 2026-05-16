# AirTrack
> **Your current MacBook keyboard is now a trackpad. Zero extra hardware, zero toggles.**

> **Seamless control powered by multimodal sensor fusion.**

*⚠️ **Note: AirTrack is currently a Work In Progress (WIP).** Core multimodal fusion models and DSP pipelines are under active development.*

AirTrack is a software defined virtual trackpad for MacBooks that transforms the physical laptop keyboard into a gesture interface / trackpad. By utilizing a **Late Sensor Fusion architecture**, it combines your webcam, built-in microphones/speakers, and OS level keystroke dynamics to perfectly distinguish between intentional typing and trackpad gestures. 

## 3 Module Architecture
1. **Action State Discriminator:** Listens to structure-borne acoustic impacts (bone conduction) and OS keystroke micro delays to instantly switch between `TYPING` and `GESTURE` modes without hotkeys.
2. **Kinematic Trajectory Decoder:** Uses top down 2D vision to translate fingertip movement into OS level cursor commands and clicks.
3. **Active Contact Estimator:** Emits an inaudible 20kHz FMCW Sonar from your MacBook speakers to measure the acoustic cross-section (Doppler shift) of your fingers, perfectly differentiating a 2-finger scroll from a 3-finger swipe without relying on camera depth.
