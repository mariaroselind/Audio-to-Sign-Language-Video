# Real-Time Audio-to-Sign Language Translation System

## Overview
This project is an AI-powered real-time Audio-to-Sign Language translation system developed to improve communication between hearing and deaf individuals.

The system converts speech into expressive sign language videos using:
- Automatic Speech Recognition (ASR)
- Text-to-Gloss Generation
- Non-Manual Marker (NMM) Generation
- Speech Emotion Recognition (SER)
- 3D Avatar Animation

The project focuses on generating more natural and meaningful sign language output by combining linguistic understanding and emotion detection.

---

# Modules

## 1. ASR Module
This module handles:
- Speech-to-Text conversion
- Text-to-Gloss generation
- Non-Manual Marker (NMM) generation

### Features
- Converts speech into text
- Generates grammar-aware gloss
- Handles tense-aware translation
- Produces facial expression markers

---

## 2. SER Module
Speech Emotion Recognition module.

### Features
- Detects emotions from speech
- Uses audio features such as:
  - MFCC
  - Pitch
  - Energy
- Uses CNN and LSTM models

---

## 3. Animation Module
This module generates sign language animation using a 3D avatar.

### Features
- Blender avatar animation
- Gesture rendering
- Facial expression rendering
- Sign language video generation

---

# System Workflow

```text
Speech Input
      ↓
ASR Module
(Speech-to-Text + Text-to-Gloss + NMM)
      ↓
SER Module
(Emotion Detection)
      ↓
Animation Module
(Blender Avatar)
      ↓
Sign Language Video Output
```

---

# Technologies Used

- Python
- NLP
- CNN
- LSTM
- Transformer Models
- Blender
- TensorFlow / PyTorch

---

# Project Structure

```text
project/
│
├── ASR_Module/
├── SER_Module/
├── Animation_Module/
│
├── images/
│
├── README.md
└── requirements.txt
```

---

# Results

## Result 1
<img width="1637" height="909" alt="1" src="https://github.com/user-attachments/assets/3b390d47-8857-4018-b259-a8b1fa133b03" />

## Result 2
<img width="1567" height="890" alt="2" src="https://github.com/user-attachments/assets/e3b93f43-4b8a-4076-8fdd-de6f549b72a2" />

## Result 3
<img width="1602" height="900" alt="3" src="https://github.com/user-attachments/assets/6f67fb2f-5502-425f-97ec-d1769f923d0f" />

## Result 4
<img width="1580" height="894" alt="4" src="https://github.com/user-attachments/assets/4c672dab-94a4-4427-b8da-9913f2fe2788" />

## Result 5
<img width="1604" height="1017" alt="5" src="https://github.com/user-attachments/assets/a3a40e6d-64a8-4942-ae8a-71dc70acd219" />
# Installation

```bash
git clone https://github.com/your-username/audio-to-sign-language.git

cd audio-to-sign-language

pip install -r requirements.txt
```

---

# Run the Project

```bash
python app.py
```

---

# Future Improvements

- Real-time deployment
- Multi-language support
- Better avatar animation
- Improved gloss generation
- Mobile application integration

---

# Applications

- Accessibility systems
- Educational platforms
- Public communication systems
- Healthcare communication assistance

---

# Authors

- Maria Roselind
- Sangeetha K
- Siona Shaji Zachrias

---

# License

This project is developed for educational and research purposes.
