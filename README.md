# 🧠 Revon — AI Learning Disability Detection System

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 in Chrome or Edge (required for microphone).
## 🎙️ Microphone — How It Works


The Reading Test uses the Web Speech API built into Chrome/Edge.

Step 1: Paragraph is displayed in the chosen language
Step 2: Click "Start Recording" — browser asks for mic permission
Step 3: Speak the paragraph clearly
Step 4: Click "Stop Recording" — transcript appears automatically
Step 5: Click "Confirm & Calculate Accuracy"
Step 6: Python difflib.SequenceMatcher computes 0-100% accuracy

NOTE: Web Speech API requires Chrome or Edge. Firefox is NOT supported.

## Language Support

Switch between English and Hindi using the dropdown in the top-right corner.
Everything changes: buttons, labels, headings, reading paragraph, recommendations.
The mic uses en-US for English and hi-IN for Hindi automatically.

## ML Classes
Normal | Dyslexia Risk | ADHD Risk | Learning Difficulty
Validation accuracy: ~99%

## Disclaimer
Revon is a screening tool only — not a clinical diagnosis.
