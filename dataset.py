import os
import numpy as np
from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from config import EMOTIONS, LABEL_NAMES, SAMPLE_RATE, DURATION, DATA_PATHS
from preprocess import audio_to_melspec, augment_audio


# ── label parsers ──────────────────────────────────────────────────────────

def get_label_ravdess(filename):
    parts = os.path.basename(filename).replace(".wav", "").split("-")
    code_map = {
        '01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad',
        '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    return code_map.get(parts[2]) if len(parts) >= 3 else None


def get_label_crema(filename):
    parts = os.path.basename(filename).replace(".wav", "").split("_")
    code_map = {
        'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad',
        'ANG': 'angry',   'FEA': 'fearful', 'DIS': 'disgust'
    }
    return code_map.get(parts[2]) if len(parts) >= 3 else None


# ── dataset builder ────────────────────────────────────────────────────────

def build_dataset():
    """Walk all dataset folders, extract mel-spectrograms, return X and y."""
    datasets = [
        ("RAVDESS", DATA_PATHS['ravdess'], get_label_ravdess),
        ("CREMA-D", DATA_PATHS['crema'],   get_label_crema),
    ]

    X, y_labels = [], []
    dataset_counts = {}

    for ds_name, ds_path, label_fn in datasets:
        if not os.path.exists(ds_path):
            print(f"Skipping {ds_name} — path not found: {ds_path}")
            continue

        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(ds_path)
            for f in fs if f.endswith(".wav")
        ]
        count = 0
        print(f"\nProcessing {ds_name} ({len(files)} files)...")

        for fp in tqdm(files, desc=ds_name):
            label = label_fn(fp)
            if label is None or label not in EMOTIONS:
                continue
            try:
                audio, sr = librosa.load(fp, sr=SAMPLE_RATE, duration=DURATION)
                if len(audio) < sr * 0.5:
                    continue

                # original
                mel = audio_to_melspec(audio, sr)
                X.append(mel)
                y_labels.append(EMOTIONS[label])
                count += 1

                # augmented versions
                for aug_audio in augment_audio(audio, sr):
                    aug_audio = librosa.util.fix_length(aug_audio, size=int(sr * DURATION))
                    mel = audio_to_melspec(aug_audio, sr)
                    X.append(mel)
                    y_labels.append(EMOTIONS[label])
                    count += 1

            except Exception:
                continue

        dataset_counts[ds_name] = count
        print(f"  {ds_name}: {count} clips (with augmentation)")

    X        = np.array(X)[..., np.newaxis]
    y_labels = np.array(y_labels)

    print(f"\n{'='*45}")
    print(f"Total samples (with augmentation): {len(X)}")
    print("\nEmotion distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y_labels == i)
        bar   = '█' * (count // 50)
        print(f"  {name:<12}: {count:>5}  {bar}")

    return X, y_labels


# ── split, balance, normalise ──────────────────────────────────────────────

def split_and_balance(X, y_labels):
    """Train/test split → oversample minority classes → normalise."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    print("Class counts before balancing:")
    for i, n in enumerate(LABEL_NAMES):
        print(f"  {n}: {np.sum(y_train == i)}")

    n_samples  = X_train.shape[0]
    orig_shape = X_train.shape[1:]
    X_flat     = X_train.reshape(n_samples, -1)

    ros = RandomOverSampler(random_state=42)
    X_flat_bal, y_train_bal = ros.fit_resample(X_flat, y_train)
    X_train_bal = X_flat_bal.reshape(-1, *orig_shape)

    print(f"\nAfter balancing: {X_train_bal.shape[0]} samples")
    print("Class counts after balancing:")
    for i, n in enumerate(LABEL_NAMES):
        print(f"  {n}: {np.sum(y_train_bal == i)}")

    # normalise using training stats
    mean = X_train_bal.mean()
    std  = X_train_bal.std() + 1e-8
    X_train_bal = (X_train_bal - mean) / std
    X_test      = (X_test      - mean) / std

    np.save("mean.npy", mean)
    np.save("std.npy",  std)
    print(f"\nMean: {mean:.4f}  Std: {std:.4f}")

    return X_train_bal, X_test, y_train_bal, y_test