"""Local webcam recognition for trained Bhutanese sign digit/alphabet CNNs."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = {
    "digits": {
        "model": ROOT / "models" / "digits.keras",
        "labels": ROOT / "models" / "digits_labels.json",
        "title": "Bhutanese Sign Digits",
    },
    "alphabet": {
        "model": ROOT / "models" / "letters.keras",
        "labels": ROOT / "models" / "letters_labels.json",
        "title": "Bhutanese Sign Alphabet",
    },
}
IMAGE_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.70


def load_model_for(mode):
    cfg = MODEL_CONFIG[mode]
    if not cfg["model"].exists() or not cfg["labels"].exists():
        raise FileNotFoundError(
            f"Missing trained {mode} model. Train it first using the commands in README.md.\n"
            f"Expected model: {cfg['model']}\nExpected labels: {cfg['labels']}"
        )
    model = tf.keras.models.load_model(cfg["model"])
    with cfg["labels"].open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels


def preprocess(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def draw_text(frame, text, position, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    print("Bhutanese Sign Language Recognition")
    print("Press 1 = digits | 2 = alphabet | q/Esc = quit")
    mode = "digits"
    loaded = {}
    prediction_text = "Show a sign to the camera"
    confidence = 0.0
    frame_count = 0
    smooth_probs = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera 0. Close other camera apps or try camera index 1 in app.py.")
        return

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Unable to read a webcam frame.")
                break
            frame = cv2.flip(frame, 1)

            if mode not in loaded:
                try:
                    loaded[mode] = load_model_for(mode)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    prediction_text = "Model not trained yet"
                    model_error = str(exc).splitlines()[0]
                else:
                    model_error = ""

            frame_count += 1
            if mode in loaded and frame_count % 2 == 0:
                model, labels = loaded[mode]
                probs = model.predict(preprocess(frame), verbose=0)[0]
                smooth_probs.append(probs)
                smooth_probs = smooth_probs[-5:]
                avg_probs = np.mean(smooth_probs, axis=0)
                best = int(np.argmax(avg_probs))
                confidence = float(avg_probs[best])
                label = str(labels.get(str(best), best))
                prediction_text = label if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (20, 25, 35), -1)
            frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
            draw_text(frame, MODEL_CONFIG[mode]["title"], (20, 32), 0.75, (120, 220, 255))
            draw_text(frame, f"Prediction: {prediction_text}", (20, 72), 0.9)
            draw_text(frame, f"Confidence: {confidence:.1%}", (20, 105), 0.58)
            draw_text(frame, "1 Digits  |  2 Alphabet  |  Q Quit", (20, frame.shape[0] - 20), 0.65, (230, 230, 230))

            if mode not in loaded:
                draw_text(frame, "Train the selected model first (see README.md)", (20, 155), 0.6, (80, 180, 255))
                if 'model_error' in locals():
                    draw_text(frame, model_error[:100], (20, 185), 0.45, (80, 180, 255))

            cv2.imshow("Bhutanese Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("1") and mode != "digits":
                mode = "digits"
                smooth_probs = []
                prediction_text = "Show a sign to the camera"
                confidence = 0.0
            elif key == ord("2") and mode != "alphabet":
                mode = "alphabet"
                smooth_probs = []
                prediction_text = "Show a sign to the camera"
                confidence = 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
