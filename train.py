import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vocab        import Vocabulary
from torch_dataset import GlossDataset
from model        import GlossTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train():

    # --- Load dataset ---
    dataset_path = os.path.join(BASE_DIR, "dataset.json")
    with open(dataset_path, "r") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} training samples.")

    # --- Build vocabularies ---
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build([s["input"]  for s in samples])
    tgt_vocab.build([s["output"] for s in samples])

    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")

    # --- Prepare dataset and dataloader ---
    dataset    = GlossDataset(samples, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on       : {device}\n")

    # --- Model ---
    model = GlossTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD token

    # --- Training loop ---
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_in  = tgt[:, :-1]  # decoder input  (drop last token)
            tgt_out = tgt[:, 1:]   # expected output (drop SOS token)

            logits = model(src, tgt_in)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1:>3}/{epochs}] | Loss: {avg_loss:.4f}")

    # --- Save model ---
    model_path = os.path.join(BASE_DIR, "gloss_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # --- Save vocabs as plain dicts (compatible with PyTorch 2.6+) ---
    vocab_path = os.path.join(BASE_DIR, "vocabs.pt")
    torch.save({
        "src_token2idx": src_vocab.token2idx,
        "src_idx2token": src_vocab.idx2token,
        "tgt_token2idx": tgt_vocab.token2idx,
        "tgt_idx2token": tgt_vocab.idx2token,
    }, vocab_path)
    print(f"Vocabs saved to : {vocab_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    train()