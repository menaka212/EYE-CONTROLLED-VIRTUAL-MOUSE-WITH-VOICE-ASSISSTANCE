🧠 Eye-Controlled Mouse System

An AI-powered hands-free mouse control application that uses eye gaze tracking, blinks, and head movements to control the computer cursor in real-time.
This project is designed to support motor-impaired users, provide touchless interaction, and demonstrate advanced computer vision + human-computer interaction concepts.

🎯 Project Overview

The Eye-Controlled Mouse System tracks the user’s eye movements using a standard webcam and converts them into mouse actions such as:

Cursor movement

Left click via blink detection

Automatic dwell clicking

Page scrolling via head roll detection

Smooth cursor stabilization

Voice feedback for clicks and actions

The entire system runs on Python, MediaPipe FaceMesh, and OpenCV, without any specialised hardware.

🚀 Key Features
🔹 Real-Time Eye Tracking

Uses MediaPipe FaceMesh to detect:

Iris center

Eye corners

Eye aspect ratio (EAR)

Head orientation

🔹 Blink-Based Clicking

A natural blink triggers a safe, debounce-protected left mouse click.

🔹 Dwell Click (Hands-Free Click)

If the cursor stays in the same area for a specific duration (default: 800 ms), a click is automatically performed.

🔹 Gaze-Based Cursor Control

Your eye movement directly controls the mouse.
Includes:

Customizable sensitivity

Smoothing filter to reduce jitter

Neutral calibration for accuracy

🔹 Head Roll Scrolling

Tilting your head left or right triggers:

Scroll up

Scroll down

Useful for reading long pages without using hands.

🔹 Voice Feedback (pyttsx3)

The system speaks:

“Click”

“Calibrated”

“Voice on/off”

“Dwell on/off”

“Reset”

🛠️ Tech Stack

Python 3.10+

OpenCV

MediaPipe FaceMesh

NumPy

pyautogui (cursor & click control)

pyttsx3 (text-to-speech)

Math & Geometry utilities for EAR & roll detection

🔧 How It Works

Webcam captures live video.

MediaPipe FaceMesh extracts:

468 face points

Iris landmarks

Eye region geometry

Iris position is normalized → mapped to screen coordinates.

Moving average smoothing stabilizes cursor.

Blinks and dwell-timer are used for clicking.

Head roll angle controls scrolling.

pyautogui executes mouse actions.

🧩 Use Cases

Assistive technology for people with motor disabilities

Hands-free computer interaction

Robotics control

AR/VR gaze-based interfaces

Gesture-free accessibility tools

Human-computer interaction (HCI) research

📸 Features in Action

(You can add screenshots or short demo GIF here.)

📦 Installation
pip install opencv-python mediapipe numpy pyautogui pyttsx3


Run:

python eye_mouse.py

🏁 Controls
Action	Trigger
Calibrate neutral	Press C
Toggle voice	Press V
Toggle dwell click	Press D
Reset smoothing	Press R
Quit	ESC or Q
🧬 Future Enhancements (Optional to add)

Right-click via long blink

Double-click via fast blink sequence

Visual GUI overlay for gaze indicator

Adaptive sensitivity calibration

Multi-monitor support

⭐ Why This Project Is Unique

Unlike many eye-tracking tools requiring:

Infrared cameras

Expensive hardware

Special sensors

This system works using only a webcam + Python, making it affordable and accessible.
