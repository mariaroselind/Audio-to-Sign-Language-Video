def format_semantic_tokens(semantics: dict) -> str:
    
    parts = []

    # Order matters — this mirrors natural sign language word order
    if semantics.get("TIME"):
        parts.append(f"<TIME> {semantics['TIME']}")

    if semantics.get("AGENT"):
        parts.append(f"<AGENT> {semantics['AGENT']}")

    if semantics.get("NEGATION"):
        parts.append("<NEG> TRUE")

    if semantics.get("ACTION"):
        parts.append(f"<ACTION> {semantics['ACTION']}")

    if semantics.get("PATIENT"):
        parts.append(f"<PATIENT> {semantics['PATIENT']}")

    if semantics.get("LOCATION"):
        parts.append(f"<LOCATION> {semantics['LOCATION']}")

    if semantics.get("QUESTION_WORD"):
        parts.append(f"<QW> {semantics['QUESTION_WORD']}")

    if semantics.get("TYPE"):
        parts.append(f"<TYPE> {semantics['TYPE'].upper()}")

    return " ".join(parts)