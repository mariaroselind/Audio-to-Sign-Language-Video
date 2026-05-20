import numpy as np

vocab = {}

file_path = "C:/lab/ml_pipeline/archive/wlasl_class_list.txt"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) >= 2:
            idx = int(parts[0])
            word = " ".join(parts[1:])   # handles "ice cream", etc.
            vocab[word] = idx

print("Loaded words:", len(vocab))
print("Sample:", list(vocab.items())[:10])

# Save in your main folder
np.save("gloss_vocab.npy", vocab)

print("✅ gloss_vocab.npy created successfully!")