import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from config import LABEL_NAMES
from dataset import build_dataset, split_and_balance
from model import build_cnn_lstm


def main():
    # ── build dataset ──────────────────────────────────────────────────────
    X, y_labels = build_dataset()
    X_train_bal, X_test, y_train_bal, y_test = split_and_balance(X, y_labels)

    # ── save test set for evaluate.py ──────────────────────────────────────
    np.save("X_test.npy",  X_test)
    np.save("y_test.npy",  y_test)

    # ── class weights ──────────────────────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_bal),
        y=y_train_bal
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", {LABEL_NAMES[k]: round(v, 2) for k, v in class_weight_dict.items()})

    # ── build model ────────────────────────────────────────────────────────
    model = build_cnn_lstm(num_classes=len(LABEL_NAMES))
    model.summary()

    # ── callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5', monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # ── train ──────────────────────────────────────────────────────────────
    history = model.fit(
        X_train_bal, y_train_bal,
        epochs=150,
        batch_size=32,
        validation_split=0.15,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nBest val accuracy: {max(history.history['val_accuracy']) * 100:.1f}%")


if __name__ == "__main__":
    main()