def postprocess_gloss(raw_gloss: str) -> str:
    tokens = raw_gloss.strip().upper().split()
    tokens = remove_duplicates(tokens)
    tokens = remove_special_tags(tokens)
    tokens = remove_unknown(tokens)
    return " ".join(tokens)

def remove_duplicates(tokens: list[str]) -> list[str]:
    # Remove consecutive duplicate tokens only
    # "I GO GO COLLEGE" → "I GO COLLEGE"
    # "I GO COLLEGE GO" → unchanged (non-consecutive)
    result = []
    for token in tokens:
        if not result or token != result[-1]:
            result.append(token)
    return result

def remove_special_tags(tokens: list[str]) -> list[str]:
    # Strip any leaked semantic tags from model output
    TAGS = {"<AGENT>", "<ACTION>", "<PATIENT>", "<LOCATION>",
            "<TIME>", "<NEG>", "<QW>", "<TYPE>", "<PAD>", "<SOS>", "<EOS>"}
    return [t for t in tokens if t not in TAGS]

def remove_unknown(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t != "<UNK>"]