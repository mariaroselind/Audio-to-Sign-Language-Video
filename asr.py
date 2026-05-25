import os
import whisper


def load_model(model_size: str = "small") -> whisper.Whisper:
    return whisper.load_model(model_size)


def transcribe_audio(
    audio_path: str,
    model: whisper.Whisper = None,
    model_size: str = "small",
    language: str = "en"
) -> dict:

    if not os.path.exists(audio_path):
        return {
            "text": "",
            "language": "en",
            "segments": [],
            "error": f"File not found: {audio_path}"
        }

    valid_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    if not audio_path.lower().endswith(valid_extensions):
        return {
            "text": "",
            "language": "en",
            "segments": [],
            "error": "Unsupported audio format"
        }

    if model is None:
        model = load_model(model_size)

    try:
        result = model.transcribe(
            audio_path,
            language=language,
            fp16=False,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False
        )

    except Exception as e:
        return {
            "text": "",
            "language": "en",
            "segments": [],
            "error": str(e)
        }

    text = result["text"].strip()

    if not text or len(text) < 3:
        return {
            "text": "",
            "language": result.get("language", "en"),
            "segments": [],
            "error": "empty_or_silent_audio"
        }

    return {
        "text": text,
        "language": result["language"],
        "segments": result["segments"]
    }


if __name__ == "__main__":

    TEST_AUDIO = "D:/PROJECT/MINIPROJECT/New/test.mp3"

    print("Loading Whisper model...")
    model = load_model("small")

    print(f"Transcribing: {TEST_AUDIO}")

    result = transcribe_audio(TEST_AUDIO, model=model)

    if "error" in result:
        print(f"Error: {result['error']}")

    else:
        print(f"Text     : {result['text']}")
        print(f"Language : {result['language']}")
        print(f"Segments : {len(result['segments'])}")

        for seg in result["segments"]:
            print(
                f"[{seg['start']:.2f}s → "
                f"{seg['end']:.2f}s] "
                f"{seg['text'].strip()}"
            )