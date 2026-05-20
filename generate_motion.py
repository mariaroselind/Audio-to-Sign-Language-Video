import torch
import torch.nn as nn
import numpy as np
import json
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LEN = 64
NUM_BONES = 33
D_MODEL = 256
NHEAD = 8
N_LAYERS = 4
DROPOUT = 0.1

# ---------------- MODEL ----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=TARGET_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GlossToMotion(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.gloss_emb = nn.Embedding(vocab_size, D_MODEL)
        self.frame_in = nn.Linear(NUM_BONES * 3, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)

        decoder_layer = nn.TransformerDecoderLayer(
            D_MODEL, NHEAD, D_MODEL * 4, DROPOUT, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, N_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, NUM_BONES * 3)

    def generate(self, gloss_id):
        self.eval()
        gid = torch.tensor([gloss_id], device=DEVICE)
        mem = self.gloss_emb(gid).unsqueeze(1).expand(1, TARGET_LEN, D_MODEL)

        generated = torch.zeros(1, 1, NUM_BONES * 3, device=DEVICE)

        for _ in range(TARGET_LEN - 1):
            emb = self.pos_enc(self.frame_in(generated))
            T_q = emb.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(T_q).to(DEVICE)

            out = self.decoder(emb, mem[:, :T_q], tgt_mask=mask)
            next_frame = self.out_proj(out[:, -1:])
            generated = torch.cat([generated, next_frame], dim=1)

        return generated.squeeze(0).cpu().numpy()

# ---------------- LOAD ----------------

vocab = np.load("gloss_vocab.npy", allow_pickle=True).item()

model = GlossToMotion(len(vocab)).to(DEVICE)
model.load_state_dict(torch.load("sign_model.pt", map_location=DEVICE))
model.eval()

# ---------------- INPUT ----------------

with open("input.json") as f:
    data = json.load(f)

glosses = data["glosses"]

# ---------------- GENERATE ----------------

all_motion = []

for word in glosses:
    word = word.upper()

    if word not in vocab:
        print(f"❌ {word} not in vocab")
        continue

    gid = vocab[word]

    with torch.no_grad():
        motion = model.generate(gid)

    motion = motion.reshape(TARGET_LEN, NUM_BONES, 3)

    all_motion.append(motion)

# ---------------- COMBINE ----------------

if len(all_motion) == 0:
    print("No valid glosses")
    exit()

final_motion = np.concatenate(all_motion, axis=0)

np.save("output_motion.npy", final_motion)

print("✅ Full sentence motion generated!")
print("Frames:", final_motion.shape[0])