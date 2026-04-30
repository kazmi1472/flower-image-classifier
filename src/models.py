"""Two model factories: a small from-scratch CNN as the baseline, and a
MobileNetV2-based transfer model as the strong one."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(image_size: int, num_classes: int, dropout: float = 0.4) -> tf.keras.Model:
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs

    for filters in (32, 64, 96, 128):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return models.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(
    image_size: int,
    num_classes: int,
    backbone: str = "MobileNetV2",
    dense_units: int = 256,
    dropout: float = 0.4,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Returns (full_model, backbone). Caller keeps the backbone reference so
    they can selectively unfreeze layers for fine-tuning."""
    backbones = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
    }
    if backbone not in backbones:
        raise ValueError(f"Unknown backbone {backbone!r}. Choices: {list(backbones)}")

    base = backbones[backbone](
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    preprocess = {
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    }[backbone]

    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    # Our pipeline gives images in [0,1]; the keras applications preprocessors
    # expect [0,255]. Easiest to scale back up here.
    x = layers.Lambda(lambda t: t * 255.0, name="rescale_back_to_255")(inputs)
    x = layers.Lambda(preprocess, name="backbone_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = models.Model(inputs, outputs, name=f"transfer_{backbone}")
    return model, base


def unfreeze_top(base: tf.keras.Model, n_layers: int) -> None:
    """Make the last n_layers of the backbone trainable. BN layers stay
    frozen — small batch + small dataset destabilises BN running stats."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
