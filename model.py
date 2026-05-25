# model.py
import torch
import torch.nn as nn

class GlossTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, max_len=30):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        positions = torch.arange(src.size(1), device=src.device).unsqueeze(0)

        src_emb = self.src_embed(src) + self.pos_embed(positions)
        tgt_emb = self.tgt_embed(tgt) + self.pos_embed(
            torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        )

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=src.device
        )
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc_out(out)