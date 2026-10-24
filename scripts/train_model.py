"""Train a small CNN on image folders, one subfolder per class."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_paths(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least two class subfolders in {data_dir}. "
            "Example: data/digits/0/image.jpg and data/digits/1/image.jpg"
        )
    paths, labels = [], []
    classes = [p.name for p in class_dirs]
    for label_id, folder in enumerate(class_dirs):
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
        if len(images) < 2:
            print(f"Warning: class '{folder.name}' has only {len(images)} images.")
        paths.extend(str(p) for p in images)
        labels.extend([label_id] * len(images))
    if len(paths) == 0:
        raise ValueError(f"No image files found under {data_dir}")
    return paths, labels, classes


def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.int32)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Root folder with one subfolder per class")
    parser.add_argument("--output", required=True, help="Output Keras model path")
    parser.add_argument("--labels", required=True, help="Output JSON labels path")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    paths, labels, classes = collect_paths(Path(args.data))
    counts = {name: labels.count(i) for i, name in enumerate(classes)}
    print(f"Classes: {classes}")
    print(f"Images: {len(paths)}")
    print(f"Per-class image counts: {counts}")

    try:
        train_p, val_p, train_y, val_y = train_test_split(
            paths, labels, test_size=0.20, random_state=SEED, stratify=labels
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified validation split. Add more images per class."
        ) from exc

    train_ds = make_dataset(train_p, train_y, training=True)
    val_ds = make_dataset(val_p, val_y, training=False)
    model = build_model(len(classes))
    model.summary()

    output = Path(args.output)
    labels_path = Path(args.labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output), monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )
    model.save(output)
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump({str(i): label for i, label in enumerate(classes)}, f, indent=2)

    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        probabilities = model.predict(batch_x, verbose=0)
        y_pred.extend(probabilities.argmax(axis=1).tolist())
        y_true.extend(batch_y.numpy().tolist())

    print("\nValidation classification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.65), max(6, len(classes) * 0.55)))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    report_path = output.parent / f"{output.stem}_confusion_matrix.png"
    fig.savefig(report_path, dpi=160)
    plt.close(fig)

    print(f"\nSaved model: {output}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved confusion matrix: {report_path}")
    print("Note: image-level random splits may overestimate generalisation if near-duplicate frames exist.")


if __name__ == "__main__":
    main()
