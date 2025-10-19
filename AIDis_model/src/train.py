"""
AIDis Makale Parse - Model Eğitimi

Bu modül, CNN modelini eğitir ve kaydeder.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import build_cnn

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BATCH_SIZE = 64
IMG_SIZE = (28, 28)
EPOCHS = 50
SEED = 42

def main():
    train_datagen = ImageDataGenerator(rescale=1./255.0)
    val_datagen   = ImageDataGenerator(rescale=1./255.0)

    train_gen = train_datagen.flow_from_directory(
        PROC / "train",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )
    val_gen = val_datagen.flow_from_directory(
        PROC / "val",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("class_indices (train):", train_gen.class_indices)
    with open(ROOT / "class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    # Sınıf ağırlığı hesaplama (dengesiz veri için)
    y_train = train_gen.classes
    _, counts = np.unique(y_train, return_counts=True)
    if len(counts) == 2 and min(counts) > 0:
        total = counts.sum()
        cw = {0: total/(2*counts[0]), 1: total/(2*counts[1])}
        print("class_weight:", cw)
    else:
        cw = None

    model = build_cnn(input_shape=(28, 28, 1), dropout_rate=0.5)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ModelCheckpoint(str(ROOT / "best_model.keras"), monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=cw
    )

    model.save(str(ROOT / "final_model.keras"))  # Model kaydet (.keras formatında)
    print("Eğitim tamamlandı. Model kaydedildi:", ROOT / "final_model.keras")

if __name__ == "__main__":
    main()
