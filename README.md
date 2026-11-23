# PWCV

**PWCV** is an open-source, educational experiment for controlling presentations with camera-based gestures and optional voice commands. It explores how OpenCV, MediaPipe, YOLO, speech recognition, and system automation can be combined to create a hands-free presentation experience.

> **Project status:** Experimental / archived. The repository contains several development attempts and proof-of-concept applications. Features and stability vary between versions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Archived%20Prototype-orange.svg)

## Highlights

- Hand-gesture and body-pose detection
- Next/previous slide navigation through `pyautogui`
- Camera view toggle and application hotkeys
- Optional voice recognition and adaptive pronunciation support
- Object detection experiments with YOLO
- Bluetooth device detection and presentation-control experiments
- Tkinter interfaces with selectable modes and themes
- Multiple preserved implementation attempts showing the project's evolution

## Repository Structure

```text
PWCV/
├── App.py                         # Early hand-gesture controller
├── App_v*.py                      # Incremental camera and control prototypes
├── First_success/                 # Early working implementations
├── attempt_4/ ... attempt_18/     # Development iterations
├── face_detection_test/            # Face detection experiments
├── requirements.txt               # Python dependencies
└── README.md
```

The most advanced available iteration is under `attempt_18/`, with `App_v21_nearbyChangeOfLogic.py` as its main application entry point. It can require additional local assets or platform-specific modules, so some older experiments may need debugging before they run.

## Getting Started

### Prerequisites

- Windows 10 or later is recommended
- Python 3.10 or later
- A webcam
- `pip` and a virtual-environment tool
- A presentation application such as Microsoft PowerPoint
- A microphone for voice-controlled variants

Some experimental scripts use Windows-specific APIs, global keyboard hooks, Bluetooth features, or microphone access.

### Installation

Clone the repository:

```bash
git clone https://github.com/TageYassir/PWCV.git
cd PWCV
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install cvzone cx-Freeze
```

### Run an Example

The first hand-control prototype can be launched with:

```bash
python App.py
```

This prototype uses:

- **F12** to show or hide the camera window
- **Shift+F12** to exit
- Open hand gestures to move between slides

To try the latest preserved iteration:

```bash
python attempt_18/App_v21_nearbyChangeOfLogic.py
```

The application may try to use local model weights or files that are not included in the repository. When a run reports a missing file, check the corresponding script and download or provide the expected asset yourself.

## Setup Guide

A packaged setup guide is also available from the original project:

[PWCV Setup Guide](https://drive.google.com/file/d/1f7lhmSb7Ntl3cJGZJKKFV3xX3IWN5E-2/view?usp=sharing)

## Experimental Features

Depending on the selected script, PWCV explores:

- Hand and finger tracking
- Face and pose tracking
- Voice commands
- YOLO object detection
- Camera zoom and angle detection
- Standing/sitting detection
- Bluetooth presence detection
- Speech pronunciation profiles
- Presentation hotkey automation
- Tkinter desktop interfaces

## Known Limitations

- This is a collection of iterations rather than a finalized application.
- Hardware, camera, OS, and library versions can significantly affect behavior.
- Some scripts contain placeholder assets, generated profiles, or technical TODOs.
- `requirements.txt` is a broad dependency list and may include packages unused by the selected script.
- Model downloads or missing local files may be required.
- Running with elevated or unnecessary system permissions may expose system automation features; only run code you understand and trust.

## Security and Responsible Use

PWCV can control a keyboard, use a microphone, scan devices, and access a camera. Before running it:

- Review the selected script.
- Grant camera and microphone permissions only when needed.
- Avoid sharing generated profiles or recordings containing personal data.
- Do not use gesture or voice automation where unintended system input could cause harm.

## Contributing

Contributions for study, experimentation, bug fixes, documentation, and reproducible improvements are welcome.

1. Fork the repository.
2. Create a focused feature branch.
3. Document the steps needed to reproduce and test your change.
4. Open a pull request with a clear explanation.

For larger ideas or direct collaboration, contact Yassir Tagemouati through GitHub.

## Contributor

| Author | Area |
|:--:|:--|
| <img src="https://github.com/TageYassir.png" width="120"/> | **Yassir Tagemouati** <br><br> Big Data Engineering Student (Cycle d'Ingénieur) <br> Azure Data Engineering • Data Warehousing • Analytics <br><br> **GitHub:** [@TageYassir](https://github.com/TageYassir) |

## Open Source and Study Use

This project is published for learning and educational study. It is released under the [MIT License](LICENSE), so you can study, modify, and reuse it subject to the license terms.

## License

Copyright (c) 2026 Yassir Tagemouati

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
