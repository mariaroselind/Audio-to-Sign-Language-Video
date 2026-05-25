# vocab.py

class Vocabulary:
    def __init__(self):
        self.token2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def build(self, sentences: list[str]):
        for sentence in sentences:
            for token in sentence.split():
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token

    def encode(self, sentence: str) -> list[int]:
        return [self.token2idx.get(t, 3) for t in sentence.split()]

    def decode(self, indices: list[int]) -> str:
        return " ".join(self.idx2token.get(i, "<UNK>") for i in indices
                        if i not in (0, 1, 2))  # skip PAD, SOS, EOS

    def __len__(self):
        return len(self.token2idx)