import os
import torch

from asr          import load_model as load_asr_model, transcribe_audio
from preprocess   import preprocess
from linguistics  import analyze_linguistics
from semantics    import extract_semantics
from token_format import format_semantic_tokens
from infer        import load_vocabs, load_gloss_model, generate_gloss
from postprocess  import postprocess_gloss
from nmm          import generate_nmm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_pipeline(audio_path: str, emotion: str = "neutral") -> dict:
    """
    Full pipeline: audio file → gloss + NMM output.

    Args:
        audio_path : path to .wav file
        emotion    : emotion string from teammate's module
                     e.g. "happy", "sad", "angry", "neutral"

    Returns:
        {
            "transcript" : raw ASR text,
            "clean_text" : preprocessed text,
            "semantics"  : extracted semantic roles,
            "token_seq"  : formatted semantic token sequence,
            "gloss"      : final sign language gloss,
            "nmm"        : non-manual marker instructions
        }
    """

    # --- Stage 1: ASR ---
    print("[1/8] Transcribing audio...")
    asr_model  = load_asr_model("small")
    asr_result = transcribe_audio(audio_path, model=asr_model)

    if "error" in asr_result:
        print(f"ASR Error: {asr_result['error']}")
        return {}

    # --- Stage 2: Preprocessing ---
    print("[2/8] Preprocessing text...")
    clean_text = preprocess(asr_result["text"])

    # --- Stage 3: Linguistic Analysis ---
    print("[3/8] Running linguistic analysis...")
    tokens = analyze_linguistics(clean_text)

    # --- Stage 4: Semantic Extraction ---
    print("[4/8] Extracting semantics...")
    semantics = extract_semantics(tokens, clean_text)

    # --- Stage 5: Token Formatting ---
    print("[5/8] Formatting semantic tokens...")
    token_seq = format_semantic_tokens(semantics)

    # --- Stage 6: Gloss Generation ---
    print("[6/8] Generating gloss...")
    src_vocab, tgt_vocab = load_vocabs(os.path.join(BASE_DIR, "vocabs.pt"))
    gloss_model          = load_gloss_model(
                               src_vocab, tgt_vocab,
                               model_path=os.path.join(BASE_DIR, "gloss_model.pt")
                           )
    raw_gloss  = generate_gloss(token_seq, src_vocab, tgt_vocab, gloss_model)

    # --- Stage 7: Post-processing ---
    print("[7/8] Post-processing gloss...")
    final_gloss = postprocess_gloss(raw_gloss)

    # --- Stage 8: NMM Generation ---
    print("[8/8] Generating NMM markers...")
    nmm = generate_nmm(semantics, emotion)

    return {
        "transcript": asr_result["text"],
        "clean_text": clean_text,
        "semantics":  semantics,
        "token_seq":  token_seq,
        "gloss":      final_gloss,
        "nmm":        nmm
    }


if __name__ == "__main__":
    import sys

    # Pass audio path and optional emotion as command line args
    # Usage: python pipeline.py test.wav happy
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "test.mp3"
    emotion    = sys.argv[2] if len(sys.argv) > 2 else "neutral"

    print(f"\nRunning pipeline on: {audio_path}")
    print(f"Emotion input      : {emotion}\n")

    result = run_pipeline(audio_path, emotion)

    if result:
        print("\n--- FINAL OUTPUT ---")
        print(f"Transcript : {result['transcript']}")
        print(f"Clean text : {result['clean_text']}")
        print(f"Semantics  : {result['semantics']}")
        print(f"Token seq  : {result['token_seq']}")
        print(f"Gloss      : {result['gloss']}")
        print(f"NMM        : {result['nmm']}")