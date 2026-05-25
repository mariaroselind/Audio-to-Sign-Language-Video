# torch_dataset.py
import torch
from torch.utils.data import Dataset

class GlossDataset(Dataset):
    def __init__(self, samples, src_vocab, tgt_vocab, max_len=30):
        self.samples   = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src = self.samples[idx]["input"]
        tgt = self.samples[idx]["output"]

        src_ids = self.src_vocab.encode(src)
        tgt_ids = [1] + self.tgt_vocab.encode(tgt) + [2]  # SOS + tokens + EOS

        # Pad to max_len
        src_ids = src_ids[:self.max_len] + [0] * max(0, self.max_len - len(src_ids))
        tgt_ids = tgt_ids[:self.max_len] + [0] * max(0, self.max_len - len(tgt_ids))

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long)
        )