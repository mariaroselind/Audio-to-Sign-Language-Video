import spacy

nlp = spacy.load("en_core_web_sm")

def analyze_linguistics(clean_text: str) -> list[dict]:
    doc = nlp(clean_text)
    tokens = []
    for token in doc:
        tokens.append({
            "text":       token.text,
            "lemma":      token.lemma_,
            "pos":        token.pos_,       # NOUN, VERB, ADV, etc.
            "dep":        token.dep_,       # nsubj, dobj, neg, ROOT, etc.
            "head":       token.head.text,  # the word this token depends on
            "is_stop":    token.is_stop,
            "morph":      str(token.morph)  # captures Tense=Past, Tense=Fut, etc.
        })
    return tokens