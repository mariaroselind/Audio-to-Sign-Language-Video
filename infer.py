import os
import torch
from vocab import Vocabulary
from model import GlossTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_vocabs(path: str = None):
    """
    Load source and target vocabularies from saved .pt file.
    Returns (src_vocab, tgt_vocab) as Vocabulary objects.
    """
    if path is None:
        path = os.path.join(BASE_DIR, "vocabs.pt")

    data = torch.load(path, weights_only=True)

    src_vocab = Vocabulary()
    src_vocab.token2idx = data["src_token2idx"]
    src_vocab.idx2token = data["src_idx2token"]

    tgt_vocab = Vocabulary()
    tgt_vocab.token2idx = data["tgt_token2idx"]
    tgt_vocab.idx2token = data["tgt_idx2token"]

    return src_vocab, tgt_vocab


def load_gloss_model(src_vocab: Vocabulary,
                     tgt_vocab: Vocabulary,
                     model_path: str = None,
                     device: str = "cpu") -> GlossTransformer:
    """
    Load the trained GlossTransformer model from disk.
    """
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "gloss_model.pt")

    model = GlossTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab)
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    model.eval()
    return model


def generate_gloss(token_sequence: str,
                   src_vocab: Vocabulary,
                   tgt_vocab: Vocabulary,
                   model: GlossTransformer = None,
                   max_len: int = 30,
                   device: str = "cpu") -> str:
    """
    Generate gloss from a semantic token sequence string.

    Args:
        token_sequence : e.g. "<TIME> TOMORROW <AGENT> I <ACTION> GO"
        src_vocab      : source Vocabulary object
        tgt_vocab      : target Vocabulary object
        model          : pre-loaded GlossTransformer (loads from disk if None)
        max_len        : max output tokens
        device         : "cpu" or "cuda"

    Returns:
        Gloss string e.g. "TOMORROW I GO COLLEGE"
    """
    if model is None:
        model = load_gloss_model(src_vocab, tgt_vocab, device=device)

    model.eval()

    src = torch.tensor(
        [src_vocab.encode(token_sequence)],
        dtype=torch.long
    ).to(device)

    # Start with SOS token
    tgt = torch.tensor([[1]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits     = model(src, tgt)
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            if next_token == 2:  # EOS token
                break

            tgt = torch.cat([
                tgt,
                torch.tensor([[next_token]], dtype=torch.long).to(device)
            ], dim=1)

    # Decode output, skipping SOS token at index 0
    generated = tgt[0, 1:].tolist()
    return tgt_vocab.decode(generated)


if __name__ == "__main__":
    src_vocab, tgt_vocab = load_vocabs()
    model = load_gloss_model(src_vocab, tgt_vocab)

    test_input = "<TIME> TOMORROW <AGENT> I <ACTION> GO <LOCATION> COLLEGE"
    result = generate_gloss(test_input, src_vocab, tgt_vocab, model)
    print(f"Input : {test_input}")
    print(f"Gloss : {result}")