import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from config import LABEL_NAMES


def main():
    # ── load saved model and test data ────────────────────────────────────
    model  = load_model("best_model.h5")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    # ── overall accuracy ──────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.1f}%")

    # ── predictions ───────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # ── confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # ── classification report ─────────────────────────────────────────────
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))


if __name__ == "__main__":
    main()