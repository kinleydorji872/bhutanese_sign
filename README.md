# Bhutanese Sign Language Recognition

An educational prototype for exploring Bhutanese Sign Language (BhSL) digit and alphabet recognition on desktop and mobile browsers.

## Current status

- **Web UI:** Responsive upload form with image preview and digits/alphabet selection.
- **Prediction API:** `POST /api/predict` accepts a JPG, PNG, or WEBP image up to 4 MB. It validates and preprocesses the image, then calls a trained CNN when model files and TensorFlow are available.
- **Health endpoint:** `GET /api/health` reports whether the model files are present in the current runtime.
- **Model status:** This repository does not include trained model weights. Until a model is trained and made available to the inference runtime, the API returns `503 MODEL_NOT_INSTALLED` rather than inventing a prediction.
- **Local webcam:** The OpenCV program is `scripts/webcam_app.py`.
- **Dataset limitation:** The linked Kaggle dataset covers digits, not the alphabet. Alphabet recognition needs a separate, correctly labelled BhSL alphabet dataset.

Predictions are experimental and must not be treated as an authoritative translation service.

## Run the web app locally

Use Python 3.11 recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m flask --app app run --debug
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m flask --app app run --debug
```

Open http://127.0.0.1:5000. The upload form and API can be tested without a trained model; the API will clearly report that the model is missing.

## Train a digit model

Install machine-learning dependencies:

```bash
pip install -r requirements-ml.txt
```

Download the [Kaggle Bhutanese Sign Language Digits dataset](https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit). Organise it into folders by class, for example:

```text
data/
  digits/
    0/
    1/
    2/
    ...
    9/
```

Train the model:

```bash
python scripts/train_model.py --data data/digits --output models/digits.keras --labels models/digits_labels.json --epochs 15
```

The image upload and webcam pipelines both resize images to 64 x 64 RGB and normalise pixels to 0–1. Keep the preprocessing consistent with the training script.

For alphabet signs, collect a separate verified BhSL alphabet dataset and organise it into class folders such as `data/letters/A/`, `data/letters/B/`, etc. Do not assume ASL or ISL labels are interchangeable with BhSL. Then train:

```bash
python scripts/train_model.py --data data/letters --output models/letters.keras --labels models/letters_labels.json --epochs 20
```

## Test local image prediction

After training, keep the files in `models/`:

- `models/digits.keras`
- `models/digits_labels.json`
- Optionally `models/letters.keras` and `models/letters_labels.json`

Start the web app with `python -m flask --app app run --debug`, open http://127.0.0.1:5000, choose a category, upload a sign image, and click **Recognise image**.

Or test the API using curl:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "mode=digits" \
  -F "image=@sample-sign.jpg"
```

Health check:

```bash
curl http://127.0.0.1:5000/api/health
```

## Deploying actual inference

Vercel serves the web UI, but **the trained model and TensorFlow runtime are not included by default**. The hosted endpoint reports this state instead of returning a fake prediction. For production use, deploy the inference API on a Python service that can include TensorFlow and the trained weights, or explicitly verify that your Vercel project can package the required runtime and model within its function limits. Vercel Functions have a 4.5 MB request-body limit, so this app caps uploads at 4 MB. See the [Vercel function limits](https://vercel.com/docs/functions/limitations) and [Python runtime documentation](https://vercel.com/docs/functions/runtimes/python).

Do not commit participant photos, datasets, or large model weights to a public repository. Store them privately or use a suitable model/object-storage service with access controls. If model files are not included in the deployment, configure the inference service to fetch them from protected storage.

## Evaluation and responsible use

- Keep validation data separate from training data; where possible, split by signer/session.
- Report accuracy, per-class precision/recall/F1, and a confusion matrix.
- Test with people who did not contribute training images.
- Keep a confidence threshold and show “Uncertain” when confidence is low.
- Obtain consent before collecting or storing participants' images/videos.
- Check dataset licensing and attribution before redistributing data or weights.

## Sources

- Kaggle Bhutanese Sign Language Digits: https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit
- Related research repository: https://github.com/KarmaCST/Bhutanese-Sign-Language-Digits-Recognition
