# 🧠 Handwritten Digits and Alphabets Classification with Voice Feedback

A web-based AI application that recognizes handwritten digits (0–9) and alphabets (A–Z) from sketches, predicts them using a trained MLP model, and provides real-time voice feedback using gTTS.

---

## 🎯 Project Objective

- Recognize handwritten digits and letters in real-time.
- Provide instant audio feedback using Text-to-Speech (TTS).
- Create an interactive and accessible user interface via browser.

---


## 🧩 System Components

### 1. 🖌 Sketch Input (Gradio)
- Users draw characters on a digital sketchpad.
- Captures input as a PIL image.

### 2. 🧼 Image Preprocessing
- Grayscale conversion
- Inversion (if needed)
- Otsu thresholding
- Cropping and padding to MNIST-style 28×28
- Normalization and flattening

### 3. 🤖 MLP Model (Trained on MSINT)
- Recognizes digits and letters
- Model saved and loaded using `pickle`

### 4. 🔊 Text-to-Speech (gTTS)
- Converts predicted character or typed text to audio
- Outputs speech in real time

### 5. 🌐 Gradio Interface
- User-friendly GUI
- Dual input: sketchpad and text box
- Immediate visual and audio feedback

---

## 🗓️ Project Workflow

1. **Dataset Analysis**  
   - Used MSINT (MNIST + EMNIST)

2. **Environment Setup**  
   - Python 3.7+, OpenCV, NumPy, gTTS, Gradio

3. **Preprocessing Pipeline**  
   - Clean and resize images to 28×28 format

4. **Model Loading & Prediction**  
   - MLP model returns label predictions

5. **TTS & GUI Integration**  
   - Converts output into voice using gTTS

6. **Testing & Validation**  
   - Checked sketch accuracy, audio quality, and browser support

---

## 🧪 Technologies Used

| Category      | Tools/Libraries                         |
|---------------|------------------------------------------|
| Language      | Python 3.7+                              |
| ML Model      | MLP Classifier (Scikit-learn)            |
| Interface     | Gradio                                   |
| Image Handling| OpenCV, Pillow, NumPy                    |
| Audio Output  | Google Text-to-Speech (gTTS)             |
| Serialization | Pickle                                   |

---

## 💻 Hardware & Software Requirements

- **OS:** Windows 10 / Linux (Ubuntu recommended)
- **RAM:** Minimum 2GB (4GB preferred)
- **Browser:** Any modern browser
- **Python Libraries:**  
  `opencv-python`, `Pillow`, `numpy`, `scikit-learn`, `gTTS`, `gradio`, `pickle`

---

## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/handwriting-voice-ai.git
   cd handwriting-voice-ai

   Install dependencies:

2. Install Dependencies:
   ```bash 
    pip install -r requirements.txt

3. Run the application:
    ```bash
    python main.py

4. A browser window will open:

- Draw a digit or letter → Click "Recognize & Speak"
- Or type text → Click "Play Text"



##📌 Future Improvements

- Add multilingual voice support
- Improve model accuracy with CNN
- Enable handwriting on mobile devices


