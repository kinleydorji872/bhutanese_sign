# Bhutanese Sign Language Recognition — BhSL Studio

Responsive Flask web app with HTML, CSS, JavaScript, image preview/drag-and-drop, a prediction API, and a CNN training script.

## Run locally
Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-ml.txt
python app.py
```
Open http://127.0.0.1:5000.

## Train the digit model
Arrange images into class folders (`data/digits/0/`, `data/digits/1/`, etc.), then run:

```bash
python scripts/train_model.py --data data/digits --output models/digits.keras --labels models/digits_labels.json --epochs 15
```
For alphabet recognition, provide a verified alphabet dataset and train a separate model with `--data data/alphabet --output models/alphabet.keras --labels models/alphabet_labels.json`.

Dataset reference: https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit

## Vercel deployment fix
The `PYTHON_ENTRYPOINT_NOT_FOUND` error means Vercel could not detect the Flask WSGI app. The root `app.py` exports `app = Flask(...)` at module scope and `vercel.json` does not force a legacy build override. Ensure Vercel's Root Directory is the repository root and `requirements.txt` is in that same directory, then redeploy.

The interface can run without a trained model, but `/api/predict` returns a clear error until model weights and label files are installed. TensorFlow inference may need a separate Python host. Do not commit private training images or model weights to a public repository.
