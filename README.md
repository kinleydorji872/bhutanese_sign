# Bhutanese Sign Language Recognition

An educational prototype for exploring Bhutanese Sign Language (BhSL) digit and alphabet recognition.

## Current status

- **Web demo:** Flask page deployed through Vercel; it provides a browser camera preview and project status.
- **Recognition:** Not enabled in the hosted demo yet. No trained model weights are committed to this repository.
- **Local prototype:** OpenCV webcam recognition code is in `scripts/webcam_app.py`.
- **Digits vs alphabet:** The referenced Kaggle dataset is for digits only. Alphabet recognition needs a separate, correctly labelled BhSL alphabet dataset.

Predictions should not be treated as guaranteed correct or as an authoritative translation service.

## Run the hosted web demo locally

Use Python 3.11 or newer.

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

Open http://127.0.0.1:5000. The `/api/health` endpoint returns basic service status.

## Run local ML training and webcam recognition

Install the separate machine-learning dependencies:

```bash
pip install -r requirements-ml.txt
```

The Kaggle dataset referenced by this project is:
https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit

Organise digit images into class folders, for example:

```text
data/
  digits/
    0/
    1/
    2/
    ...
    9/
```

Train a digit model:

```bash
python scripts/train_model.py --data data/digits --output models/digits.keras --labels models/digits_labels.json --epochs 15
```

After training, run the webcam app:

```bash
python scripts/webcam_app.py
```

Press `1` for digits, `2` for alphabet, and `q` or `Esc` to quit. A mode will only work after its model and label JSON are present in `models/`.

## Prepare an alphabet model

Collect or obtain a separate BhSL alphabet dataset, and confirm labels with a competent BhSL signer. Do not assume American Sign Language (ASL) or Indian Sign Language (ISL) labels are interchangeable with BhSL.

Organise it into class folders:

```text
data/
  letters/
    A/
    B/
    C/
    ...
```

Then train:

```bash
python scripts/train_model.py --data data/letters --output models/letters.keras --labels models/letters_labels.json --epochs 20
```

The baseline CNN handles individual static images, not complete sign-language sentences or motion-based signs.

## Deploy on Vercel

This repository is linked to Vercel. Each push to the production branch should trigger a new deployment.

1. Open the Vercel project dashboard and select `bhutanese-sign`.
2. Open **Deployments** and wait for the newest deployment to become **Ready**.
3. Open the production domain and check the landing page and `/api/health`.
4. If a build fails, open that deployment and inspect **Build Logs**.
5. Keep `requirements.txt` lightweight for the hosted Flask page. The TensorFlow/OpenCV training stack is intentionally in `requirements-ml.txt` for local development.

The hosted page currently does not perform model inference. To enable recognition online, train and validate a model first, then implement an inference endpoint with an appropriate hosting/runtime strategy. The camera preview uses the visitor's browser; server-side OpenCV cannot directly access a visitor's webcam.

## Evaluation and responsible use

- Keep validation data separate from training data; where possible, split by signer/session.
- Report accuracy, per-class precision/recall/F1, and a confusion matrix.
- Test with people who did not contribute training images.
- Keep a confidence threshold and display uncertainty.
- Obtain consent before collecting or storing participants' images/videos.
- Check dataset licensing and attribution before redistributing data or weights.

## Sources

- Kaggle Bhutanese Sign Language Digits: https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit
- Related research repository: https://github.com/KarmaCST/Bhutanese-Sign-Language-Digits-Recognition
