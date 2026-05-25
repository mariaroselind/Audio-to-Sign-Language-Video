# nmm.py

def generate_nmm(semantics: dict, emotion: str = "neutral") -> dict:

    nmm = {
        "eyebrows":         "neutral",
        "head_movement":    "neutral",
        "facial_expression": "neutral",
        "eye_gaze":         "forward",
        "mouth":            "neutral"
    }

    # Eyebrows — driven by sentence type
    sentence_type = semantics.get("TYPE", "declarative")
    question_word = semantics.get("QUESTION_WORD")

    if sentence_type == "question":
        if question_word:
            # WH-questions (what, where, who) → furrowed brows
            nmm["eyebrows"] = "furrowed"
            nmm["head_movement"] = "slight_forward_tilt"
        else:
            # Yes/No questions → raised brows
            nmm["eyebrows"] = "raised"
            nmm["head_movement"] = "slight_forward_tilt"

    # Head movement — driven by negation
    if semantics.get("NEGATION"):
        nmm["head_movement"] = "side_to_side"  # head shake for negation

    # Facial expression — driven by emotion from teammate's module
    EMOTION_MAP = {
        "happy":    "smile",
        "sad":      "frown",
        "angry":    "tense",
        "surprised": "wide_eyes",
        "fearful":  "tense",
        "disgusted": "wrinkled_nose",
        "neutral":  "neutral"
    }
    nmm["facial_expression"] = EMOTION_MAP.get(emotion.lower(), "neutral")

    # Eye gaze — driven by sentence type
    if sentence_type == "question":
        nmm["eye_gaze"] = "direct"   # direct eye contact for questions
    elif semantics.get("LOCATION"):
        nmm["eye_gaze"] = "directional"  # gaze toward location sign space

    # Mouth — driven by action/content
    if semantics.get("NEGATION"):
        nmm["mouth"] = "pursed"      # common mouth morpheme for negation
    elif sentence_type == "question":
        nmm["mouth"] = "open_slightly"

    return nmm