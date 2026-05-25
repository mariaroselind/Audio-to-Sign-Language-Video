import re
import contractions

FILLER_WORDS = {
    "um", "uh", "ah", "er", "hmm", "hm",
    "like", "you know", "i mean", "basically",
    "literally", "actually", "so", "well",
    "right", "okay", "ok", "alright"
}

PROTECTED_WORDS = {
    "not", "never", "no", "none", "nobody",
    "nothing", "nowhere",

    "what", "where", "when",
    "who", "why", "how",

    "yesterday", "today", "tomorrow",
    "now", "later", "always", "soon",
    "ago", "morning", "evening", "night",
    "before", "after", "already",
    "still", "yet"
}

def lowercase(text: str) -> str:
    return text.lower()

def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def remove_fillers(text: str) -> str:
    words = text.split()
    cleaned = []
    i = 0
    while i < len(words):

        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i + 1]}".lower()

            if (
                bigram in FILLER_WORDS
                and bigram not in PROTECTED_WORDS
            ):
                i += 2
                continue

        word = words[i]
        word_lower = word.lower().strip(".,!?")

        if word_lower in PROTECTED_WORDS:
            cleaned.append(word)

        elif word_lower not in FILLER_WORDS:
            cleaned.append(word)

        i += 1

    return " ".join(cleaned)

def normalize_punctuation(text: str) -> str:

    text = re.sub(r"[^\w\s.?!,]", "", text)
    text = re.sub(r"\s([?.!,])", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def preprocess(raw_text: str) -> str:

    if not raw_text or not raw_text.strip():
        return ""

    text = lowercase(raw_text)
    text = expand_contractions(text)
    text = remove_fillers(text)
    text = normalize_punctuation(text)

    return text