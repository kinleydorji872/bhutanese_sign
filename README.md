# Bhutanese Sign Language Recognition

A free, local-first webcam prototype for recognising Bhutanese sign digits and alphabet classes on a laptop.

## Important dataset limitation

The Kaggle dataset supplied for this project (\`yjvision/bhutanesesigndigit\`) is a **digits dataset**. It does not provide alphabet examples. This repository includes a reusable training pipeline for both digits and letters, but a letter model will only work after you add a correctly labelled Bhutanese sign alphabet dataset and train it.

Do not interpret predictions as guaranteed correct. Performance depends on training data, lighting, camera angle, signers, and evaluation.

## Features

- Laptop webcam preview using OpenCV.
- Separate model modes: digits and alphabet.
- CNN training script for folder-organised images.
- Confidence threshold to avoid showing weak predictions as certain.
- Local inference. No cloud API or paid service is required after dependencies and model files are installed.
- Dataset and trained weights are not included.

## 1. Requirements

- Python 3.10 or 3.11 recommended.
- A working webcam.
- Internet access to install Python packages and obtain datasets/models.
- Windows, macOS, or Linux.

## 2. Create an environment and install

### Windows PowerShell

\`\`\`powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

If PowerShell blocks activation, run this once in the current terminal:
\`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass\`

### macOS / Linux

\`\`\`bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## 3. Get the digits dataset

Download the dataset from:
https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit

Or use the Kaggle CLI after setting up your Kaggle API token:

\`\`\`bash
pip install kaggle
kaggle datasets download -d yjvision/bhutanesesigndigit -p data/downloads --unzip
\`\`\`

The training script expects images sorted into class folders, for example:

\`\`\`text
data/
  digits/
    0/
      image001.jpg
    1/
      image002.jpg
    ...
    9/
      image999.jpg
\`\`\`

If the downloaded archive already has class folders, copy/organise them under \`data/digits/\`. If it uses a different structure, organise it before training. Do not mix test images into the training folders.

## 4. Train the digit model

\`\`\`bash
python scripts/train_model.py --data data/digits --output models/digits.keras --labels models/digits_labels.json --epochs 15
\`\`\`

The script uses an 80/20 stratified random split. For stronger evaluation, split by signer or original video/session if metadata is available, so near-duplicate frames do not appear in both train and validation.

## 5. Prepare and train an alphabet model

Obtain or collect a separate Bhutanese sign alphabet dataset. Use signs standardised by Bhutanese Sign Language resources and confirm labels with a competent BhSL signer. Do not assume ASL or ISL labels are interchangeable with BhSL.

Organise it as:

\`\`\`text
data/
  letters/
    A/
    B/
    C/
    ...
\`\`\`

Train with:

\`\`\`bash
python scripts/train_model.py --data data/letters --output models/letters.keras --labels models/letters_labels.json --epochs 20
\`\`\`

Only include classes with reliable, labelled training examples. This baseline CNN handles **single static images**, not full sign-language sentences or motion-based signs.

## 6. Run the webcam app

After training at least one model:

\`\`\`bash
python app.py
\`\`\`

- Press \`1\` to select digits.
- Press \`2\` to select alphabet.
- Press \`q\` or \`Esc\` to quit.
- If the selected model file is missing, the app explains how to train it.

Camera index 0 is used by default. If needed, edit \`cv2.VideoCapture(0)\` to \`1\`.

## 7. Folder structure

\`\`\`text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── scripts/
│   └── train_model.py
├── data/
│   ├── digits/       # downloaded/organised locally; not committed
│   └── letters/      # collected/organised locally; not committed
└── models/
    ├── digits.keras  # generated after training; not committed
    └── letters.keras # generated after training; not committed
\`\`\`

## 8. Evaluation and responsible use

- Report validation accuracy, per-class precision/recall/F1, and a confusion matrix before presenting results.
- Test with people who did not contribute training images.
- Keep a confidence threshold and show “Uncertain” when confidence is low.
- Camera predictions are experimental assistance, not an authoritative translation service.
- Get consent before collecting or storing images/videos of participants.
- Check dataset licence and attribution terms before redistributing data or trained weights.

## 9. Free deployment note

The webcam app runs locally on a laptop for free. A hosted server cannot directly access a visitor's laptop webcam through server-side OpenCV; a browser deployment needs a browser camera component such as WebRTC and a different inference setup.

## Dataset source

- Kaggle Bhutanese Sign Language Digits: https://www.kaggle.com/datasets/yjvision/bhutanesesigndigit
- Related research repository: https://github.com/KarmaCST/Bhutanese-Sign-Language-Digits-Recognition
