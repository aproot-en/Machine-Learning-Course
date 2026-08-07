import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes):
    """Convolutional neural network: 3 conv blocks -> dense head."""

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Normalize 0-255 to 0-1 inside the model, so inference code never
        # has to remember to do it
        layers.Rescaling(1.0 / 255),

        # Light augmentation, only active during training.
        # This is what keeps a small CNN from memorising the training set.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # Pooling instead of Flatten keeps the dense head small
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        # 1 sigmoid output for 2 classes, softmax otherwise
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32):
    """Build, train and save the model. Returns (model, history)."""

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving, keep the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "cnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'cnn_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)
