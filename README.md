# Hand Synthesizer

**Sound ON — this demo includes audio**


https://github.com/user-attachments/assets/2d548749-17c4-44de-9ad9-512576c069b0



Hand Synthesizer is a real-time hand gesture–controlled synthesizer built with **Python**, **OpenCV**, and **MediaPipe Tasks**.
It uses computer vision to track hand and finger movements and maps them to audio parameters such as pitch, volume, and effects.

---

## Tech Stack

* **Python 3.8+**
* **OpenCV** – webcam capture and visualization
* **MediaPipe Tasks API** – real-time hand landmark detection
* **NumPy** – numerical processing
* **sounddevice** – real-time audio synthesis

---

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:artinSha/hand-synthesizer.git
cd hand-synthesizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the MediaPipe hand model

This project uses the MediaPipe Tasks API, which requires a separate model file.

Download the hand landmarker model from:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Save the file as:

```
hand_landmarker.task
```

in the project root directory.

---

### 6. Run the application

```bash
python ./src/hand-synthesizer.py
```

Press **`q`** to quit the application.
