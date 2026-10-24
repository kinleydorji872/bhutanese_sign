"""Responsive BhSL web demo with an image-upload prediction endpoint.

The hosted UI can run without TensorFlow, but actual predictions require a trained
model + labels and a runtime with TensorFlow installed.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
IMAGE_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.70

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
_loaded_models = {}


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0b1220">
<title>Bhutanese Sign Language Recognition</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#131e31;--text:#eef4ff;--muted:#a8b7d1;--accent:#65d6c4;--line:#2a3a55}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(140deg,#0b1220,#142641);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text)}
main{max-width:940px;margin:auto;padding:30px 16px 60px}.tag{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.8rem}
h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.1;margin:.5rem 0 1rem}h2{line-height:1.25}.muted{color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:clamp(16px,3vw,24px);margin:20px 0}
.actions{display:flex;flex-wrap:wrap;gap:12px}button,a.button{border:0;border-radius:10px;padding:12px 16px;font:inherit;font-weight:700;cursor:pointer;text-decoration:none}
button{background:var(--accent);color:#08201e}button:disabled{opacity:.55;cursor:wait}a.button{background:#263954;color:var(--text)}
select,input[type=file]{font:inherit;color:var(--text);max-width:100%}select{background:#0b1220;border:1px solid #41536d;border-radius:10px;padding:10px;width:100%}
label{display:block;font-weight:650;margin:12px 0 6px}.upload{border:2px dashed #526a88;border-radius:14px;padding:20px;text-align:center;background:#0e1829}
.upload.drag{border-color:var(--accent);background:#142c38}.upload input{display:block;margin:12px auto;max-width:100%}
#preview{display:none;max-width:100%;max-height:320px;margin:16px auto 0;border-radius:12px}#result{margin-top:16px;white-space:pre-wrap;overflow-wrap:anywhere}
.notice{border-left:4px solid #eabf66;padding:12px 14px;background:#2b281e;border-radius:8px}
.success{border-left-color:var(--accent);background:#142c2c}.error{border-left-color:#ff8b8b;background:#321f29}
video{width:100%;max-height:360px;background:#05080e;border-radius:12px;margin-top:14px;display:none}
footer{margin-top:30px;color:var(--muted);font-size:.9rem}code{overflow-wrap:anywhere}
</style>
</head>
<body><main>
<div class="tag">Bhutan · Educational prototype</div>
<h1>Bhutanese Sign Language Recognition</h1>
<p class="muted">Upload a clear image of one sign to try the recognition pipeline. The image is processed for a prediction and is not intentionally saved by this app.</p>

<section class="panel">
<h2>Recognise a sign image</h2>
<p class="muted">Use a single, well-lit sign with the hand clearly visible. Maximum image size: 4 MB. JPG, PNG, and WEBP are supported.</p>
<form id="predictForm">
<label for="mode">Sign category</label>
<select id="mode" name="mode"><option value="digits">Digits</option><option value="alphabet">Alphabet</option></select>
<div class="upload" id="dropArea">
<strong>Choose an image or drop it here</strong>
<p class="muted">Works on desktop and mobile browsers.</p>
<input id="imageFile" name="image" type="file" accept="image/jpeg,image/png,image/webp" required>
<img id="preview" alt="Preview of the selected sign image">
</div>
<div class="actions" style="margin-top:16px"><button id="predictButton" type="submit">Recognise image</button><button id="clearButton" type="button" class="secondary" style="background:#263954;color:var(--text)">Clear</button></div>
</form>
<div id="result" class="notice" role="status" aria-live="polite">Choose an image to begin. A trained model must be installed for predictions to work.</div>
</section>

<section class="panel">
<h2>Optional camera preview</h2>
<p class="muted">This preview only checks browser camera access; it does not run live recognition.</p>
<div class="actions"><button id="startCamera" type="button">Start camera</button><button id="stopCamera" type="button" disabled style="background:#263954;color:var(--text)">Stop camera</button></div>
<video id="video" autoplay playsinline muted></video>
<p id="cameraStatus" class="muted" aria-live="polite">Camera is off.</p>
</section>

<section class="panel">
<h2>Project status</h2>
<p>This web interface includes an upload and prediction API. Actual predictions require trained model weights and a runtime with TensorFlow installed.</p>
<div class="actions"><a class="button" href="https://github.com/kinleydorji872/bhutanese_sign" target="_blank" rel="noreferrer">View source on GitHub</a><a class="button" href="https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit" target="_blank" rel="noreferrer">Digits dataset</a></div>
<p class="muted">The referenced dataset covers digits, not the alphabet. Alphabet labels must come from a separate, verified Bhutanese Sign Language dataset.</p>
</section>
<footer>Experimental educational assistance only, not an authoritative translation service. Ask permission before collecting or sharing images of people.</footer>
</main>
<script>
const form=document.getElementById('predictForm'),fileInput=document.getElementById('imageFile'),preview=document.getElementById('preview'),result=document.getElementById('result'),button=document.getElementById('predictButton'),dropArea=document.getElementById('dropArea');
function showImage(file){if(!file){preview.removeAttribute('src');preview.style.display='none';return;}if(file.size>4*1024*1024){result.className='notice error';result.textContent='That image is larger than 4 MB. Please choose a smaller image.';fileInput.value='';showImage(null);return;}const url=URL.createObjectURL(file);preview.onload=()=>URL.revokeObjectURL(url);preview.src=url;preview.style.display='block';}
fileInput.addEventListener('change',()=>showImage(fileInput.files[0]));
['dragenter','dragover'].forEach(ev=>dropArea.addEventListener(ev,e=>{e.preventDefault();dropArea.classList.add('drag');}));
['dragleave','drop'].forEach(ev=>dropArea.addEventListener(ev,e=>{e.preventDefault();dropArea.classList.remove('drag');}));
dropArea.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f){const dt=new DataTransfer();dt.items.add(f);fileInput.files=dt.files;showImage(f);}});
form.addEventListener('submit',async e=>{e.preventDefault();const f=fileInput.files[0];if(!f){result.className='notice error';result.textContent='Please choose an image first.';return;}if(f.size>4*1024*1024){result.className='notice error';result.textContent='Please choose an image smaller than 4 MB.';return;}button.disabled=true;button.textContent='Checking image…';result.className='notice';result.textContent='Uploading image and checking model availability…';try{const body=new FormData();body.append('image',f);body.append('mode',document.getElementById('mode').value);const response=await fetch('/api/predict',{method:'POST',body});const data=await response.json();if(!response.ok)throw new Error(data.error||'Prediction failed.');result.className='notice success';result.textContent='Prediction: '+data.prediction+'\nConfidence: '+(data.confidence*100).toFixed(1)+'%\nCategory: '+data.mode+'\n'+(data.message||'');}catch(err){result.className='notice error';result.textContent=err.message+'\nIf the model has not been trained and installed, follow the README instructions.';}finally{button.disabled=false;button.textContent='Recognise image';}});
document.getElementById('clearButton').addEventListener('click',()=>{form.reset();showImage(null);result.className='notice';result.textContent='Choose an image to begin. A trained model must be installed for predictions to work.';});
const video=document.getElementById('video'),start=document.getElementById('startCamera'),stop=document.getElementById('stopCamera'),cameraStatus=document.getElementById('cameraStatus');let stream=null;
start.addEventListener('click',async()=>{try{if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('Camera access is not supported in this browser.');stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});video.srcObject=stream;video.style.display='block';start.disabled=true;stop.disabled=false;cameraStatus.textContent='Camera preview is running. Upload a still image above to use the prediction endpoint.';}catch(e){cameraStatus.textContent='Could not start camera: '+e.message+' Check permissions and use HTTPS.';}});
stop.addEventListener('click',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.style.display='none';start.disabled=false;stop.disabled=true;cameraStatus.textContent='Camera is off.';});
window.addEventListener('pagehide',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());});
</script></body></html>
"""


def _get_model(mode: str):
    """Load a model once per warm process; never trust a client-supplied path."""
    config = MODEL_CONFIG[mode]
    model_path, labels_path = config["model"], config["labels"]
    if not model_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"The trained {mode} model is not installed. Expected {model_path.name} and {labels_path.name}."
        )
    if mode not in _loaded_models:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed in this web runtime. Install the ML dependencies on a suitable inference host."
            ) from exc
        model = tf.keras.models.load_model(model_path)
        with labels_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        if not isinstance(labels, dict) or not labels:
            raise ValueError("The model label file is empty or invalid.")
        _loaded_models[mode] = (model, labels)
    return _loaded_models[mode]


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "prediction_api": "available",
        "trained_models": {
            mode: config["model"].is_file() and config["labels"].is_file()
            for mode, config in MODEL_CONFIG.items()
        },
    })


@app.post("/api/predict")
def predict():
    mode = request.form.get("mode", "digits").lower()
    if mode not in MODEL_CONFIG:
        return jsonify({"error": "Choose either digits or alphabet mode."}), 400
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please attach an image file."}), 400

    raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image must be 4 MB or smaller."}), 413

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                return jsonify({"error": "Unsupported image type. Use JPG, PNG, or WEBP."}), 415
            if source.width * source.height > 20_000_000:
                return jsonify({"error": "Image dimensions are too large. Please resize it first."}), 413
            image = source.convert("RGB").resize(IMAGE_SIZE)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "The file is not a valid supported image."}), 400

    try:
        model, labels = _get_model(mode)
    except FileNotFoundError as exc:
        return jsonify({
            "error": str(exc),
            "code": "MODEL_NOT_INSTALLED",
            "next_step": "Train the model and make its .keras and labels JSON files available to the inference runtime.",
        }), 503
    except RuntimeError as exc:
        return jsonify({"error": str(exc), "code": "INFERENCE_RUNTIME_MISSING"}), 503
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        app.logger.exception("Could not load the %s model", mode)
        return jsonify({"error": f"Could not load the trained model: {exc}"}), 500

    try:
        import numpy as np
        array = np.asarray(image, dtype="float32") / 255.0
        batch = np.expand_dims(array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        label = str(labels.get(str(best), best))
        prediction = label if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"
        return jsonify({
            "prediction": prediction,
            "class_label": label,
            "confidence": confidence,
            "mode": mode,
            "message": "Experimental result; verify with a Bhutanese Sign Language expert.",
        })
    except Exception:
        app.logger.exception("Inference failed for %s image", mode)
        return jsonify({"error": "The model could not process this image. Check the model input shape and label mapping."}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image upload is too large. Please use an image under 4 MB."}), 413
