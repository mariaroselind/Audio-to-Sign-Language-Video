def extract_semantics(tokens: list[dict], clean_text: str) -> dict:
    
    TIME_WORDS = {"yesterday", "today", "tomorrow", "now", "later", 
                  "always", "never", "soon", "ago", "morning", "evening"}
    
    LOCATION_PREPS = {"to", "at", "in", "from", "near"}
    
    QUESTION_WORDS = {"what", "where", "when", "who", "why", "how"}

    semantics = {
        "AGENT":         None,
        "ACTION":        None,
        "PATIENT":       None,
        "LOCATION":      None,
        "TIME":          None,
        "NEGATION":      False,
        "TYPE":          "declarative",
        "QUESTION_WORD": None
    }

    # Detect sentence type
    text_lower = clean_text.strip()
    if text_lower.endswith("?"):
        semantics["TYPE"] = "question"

    for token in tokens:
        t   = token["text"].lower()
        dep = token["dep"]
        pos = token["pos"]
        lemma = token["lemma"].upper()

        # AGENT — subject of main verb
        if dep == "nsubj" and semantics["AGENT"] is None:
            semantics["AGENT"] = lemma

        # ACTION — root verb, use lemma for base form
        if dep == "ROOT" and pos in ("VERB", "AUX"):
            semantics["ACTION"] = lemma

        # PATIENT — direct object
        if dep == "dobj" and semantics["PATIENT"] is None:
            semantics["PATIENT"] = lemma

        # NEGATION
        if dep == "neg":
            semantics["NEGATION"] = True

        # TIME — time adverbs or nouns
        if t in TIME_WORDS:
            semantics["TIME"] = lemma

        # LOCATION — noun following a location preposition
        if dep == "pobj" and token["head"].lower() in LOCATION_PREPS:
            semantics["LOCATION"] = lemma

        # QUESTION WORD
        if t in QUESTION_WORDS:
            semantics["QUESTION_WORD"] = lemma
            semantics["TYPE"] = "question"

    return semantics