# Bhutanese Sign Language Recognition — BhSL Studio

Responsive sign-image recognition website with a camera capture workflow for laptops and phones. Frontend and backend are kept in separate files/folders.

## Project structure

- `templates/index.html` — HTML page structure
- `static/css/style.css` — all presentation and responsive styling
- `static/js/app.js` — camera permission, live preview, capture, upload and API calls
- `app.py` — Flask server, webpage/API routes and request validation
- `scripts/train_model.py` — model training utility
- `models/` — place trained model files here (not committed)

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-ml.txt
python app.py
```

Open http://127.0.0.1:5000. The browser will ask for camera permission when you click **Start camera**. Camera access works on localhost during development and on HTTPS in production; it generally does not work on an ordinary non-HTTPS network address.

## Train the digit model

Arrange images into class folders, for example `data/digits/0/`, `data/digits/1/`, etc.

```bash
python scripts/train_model.py --data data/digits --output models/digits.keras --labels models/digits_labels.json --epochs 15
```

Alphabet mode requires its own labeled dataset and model files named `models/alphabet.keras` and `models/alphabet_labels.json`.

Dataset reference: https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit

## API

- `GET /` — frontend page
- `GET /api/health` — reports whether model files exist
- `POST /api/predict` — accepts multipart fields `image` and `mode` (`digits` or `alphabet`)

The camera only captures an image in the browser after the user presses **Capture photo**. It sends that captured image to the same prediction API as an uploaded image. No continuous video is sent to the server.

## Vercel

The root `app.py` exports a top-level Flask `app`. The root `vercel.json` avoids the legacy custom-build override. Ensure Vercel's Root Directory is the repository root and redeploy after each commit.

The interface can deploy without model weights, but predictions will return a clear error until trained model files and labels are added. TensorFlow inference may need a separate Python inference host depending on deployment runtime limits.
