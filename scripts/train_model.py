"""Train a CNN from images stored in one folder per class."""
import argparse, json
from pathlib import Path
import tensorflow as tf

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/digits")
    parser.add_argument("--output",default="models/digits.keras")
    parser.add_argument("--labels",default="models/digits_labels.json")
    parser.add_argument("--epochs",type=int,default=15)
    args=parser.parse_args(); data=Path(args.data)
    if not data.is_dir() or sum(p.is_dir() for p in data.iterdir())<2: raise SystemExit("Dataset needs at least two class subfolders, e.g. data/digits/0 and data/digits/1.")
    train=tf.keras.utils.image_dataset_from_directory(data,validation_split=.2,subset="training",seed=42,image_size=(64,64),batch_size=32)
    val=tf.keras.utils.image_dataset_from_directory(data,validation_split=.2,subset="validation",seed=42,image_size=(64,64),batch_size=32)
    labels=train.class_names; prep=lambda x,y:(tf.cast(x,tf.float32)/255.0,y)
    train=train.map(prep).prefetch(tf.data.AUTOTUNE); val=val.map(prep).prefetch(tf.data.AUTOTUNE)
    model=tf.keras.Sequential([tf.keras.layers.Input((64,64,3)),tf.keras.layers.Conv2D(32,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(64,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dropout(.3),tf.keras.layers.Dense(len(labels),activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); model.fit(train,validation_data=val,epochs=args.epochs)
    out,label_file=Path(args.output),Path(args.labels); out.parent.mkdir(parents=True,exist_ok=True); label_file.parent.mkdir(parents=True,exist_ok=True)
    model.save(out); label_file.write_text(json.dumps(labels,indent=2),encoding="utf-8"); print(f"Saved {out} and {label_file}")
if __name__=="__main__": main()
