import json
import os

from tensorflow import keras
from tensorflow.keras import layers

# This is the VGG16 layout: 13 conv layers as 2-2-3-3-3.
VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]

# Lighter variant for CPU training / small datasets.
VGG_SMALL = [(32, 2), (64, 2), (128, 3), (256, 3)]

def vgg_block(model, filters, n_conv):
    """n_conv x (Conv3x3 -> BatchNorm) then halve the resolution."""

    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(2))


def build_model(input_shape, num_classes, blocks=VGG16):

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))

    # Scale 0-255 to 0-1 inside the model, so inference cannot forget to
    model.add(layers.Rescaling(1.0 / 255))

    # Augmentation, active during fit() only
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=50, batch_size=32, blocks=VGG16):

    model = build_model(X_train.shape[1:], num_classes, blocks)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
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

        model.save(os.path.join(output_dir, "vgg_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'vgg_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)
