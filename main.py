import io
import pickle
import numpy as np
import cv2
from PIL import Image, ImageOps
from gtts import gTTS
import gradio as gr

# ——— Improved Preprocessing Function ———
def preprocess(img: Image.Image):
    # 1) Grayscale & invert if needed
    img_gray = ImageOps.grayscale(img)
    if np.mean(img_gray) > 128:
        img_gray = ImageOps.invert(img_gray)
    arr = np.array(img_gray)

    # 2) Binarize (Otsu), thin via erosion
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thin = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)

    # 3) Crop to content
    coords = cv2.findNonZero(thin)
    x,y,w,h = cv2.boundingRect(coords)
    digit = thin[y:y+h, x:x+w]

    # 4) Resize to 18×18, pad to 28×28
    if w > h:
        new_w, new_h = 18, int(18 * (h/w))
    else:
        new_h, new_w = 18, int(18 * (w/h))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, bottom = (28-new_h)//2, 28-new_h-(28-new_h)//2
    left, right = (28-new_w)//2, 28-new_w-(28-new_w)//2
    final_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=0)

    # 5) Flatten & normalize
    flat = final_img.astype(np.float32).reshape(1, -1) / 255.0
    return flat

# ——— Load the Pre-trained Model ———
with open("model/best_model_mlp.pkl", "rb") as f:
    model = pickle.load(f)

# ——— Drawing → Prediction + TTS ———
def recognize_and_speak(sketch_dict):
    img = sketch_dict["composite"]               # PIL image
    x = preprocess(img)
    pred = model.predict(x)[0]
    label = str(pred) if pred < 10 else chr(pred)

    # gTTS → raw bytes
    tts = gTTS(label)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio_bytes = buf.read()
    return label, audio_bytes

# ——— Text → TTS ———
def text_to_speech(text):
    if text.strip() == "":
        return None, None
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return text, buf.read()

# ——— Gradio App ———
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Draw a digit (0–9) or letter (A–Z)")
    with gr.Row():
        sketch = gr.Sketchpad(type="pil", image_mode="RGB")
        output_txt = gr.Textbox(label="Prediction")
    output_audio = gr.Audio(label="Spoken Result")
    btn = gr.Button("Recognize & Speak")
    btn.click(fn=recognize_and_speak,
              inputs=sketch,
              outputs=[output_txt, output_audio])

    gr.Markdown("## 📝 Type text to hear it spoken")
    text_input = gr.Textbox(label="Enter text here")
    play_btn  = gr.Button("Play Text")
    text_audio = gr.Audio(label="Text-to-Speech Output")
    play_btn.click(fn=text_to_speech,
                   inputs=text_input,
                   outputs=[text_input, text_audio])

    gr.Markdown("""
**How it works**  
- **Draw** on the sketchpad → click **Recognize & Speak** → see & hear the model’s prediction.  
- **Type** any text below → click **Play Text** → hear your text spoken via GTTS.
""")

if __name__ == "__main__":
    demo.launch()
