import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model

from config import INPUT_SHAPE, NUM_CLASSES


def build_cnn_lstm(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    CNN-LSTM hybrid:
      - CNN blocks extract local spectral/spatial features
      - Bidirectional LSTM layers learn temporal patterns
      - Dense head outputs emotion class probabilities
    """
    inp = Input(shape=input_shape)

    # ── CNN block 1 ────────────────────────────────────────────────────────
    x = layers.Conv2D(64,  (3, 3), activation='relu', padding='same')(inp)
    x = layers.Conv2D(64,  (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # ── CNN block 2 ────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── CNN block 3 ────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── CNN block 4 ────────────────────────────────────────────────────────
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # ── reshape for LSTM: (time_steps, features) ───────────────────────────
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # ── Bidirectional LSTM ─────────────────────────────────────────────────
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    # ── classifier head ────────────────────────────────────────────────────
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model