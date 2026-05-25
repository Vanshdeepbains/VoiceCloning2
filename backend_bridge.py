import requests
import uuid

BASE = "http://127.0.0.1:8000"


# ---------------- TTS ----------------
def TextToSpeech(text, lang, voice_type, speed, pitch, emotion):

    r = requests.post(
        f"{BASE}/tts",
        data={"text": text, "lang": lang}
    )

    path = f"outputs/{uuid.uuid4().hex}.mp3"
    open(path, "wb").write(r.content)

    return path


# ---------------- VOICE CLONE ----------------
def VoiceClone(text, lang, sample_path, speed, pitch, emotion):

    with open(sample_path, "rb") as f:
        r = requests.post(
            f"{BASE}/clone",
            files={"file": f},
            data={"text": text, "lang": lang}
        )

    path = f"outputs/{uuid.uuid4().hex}.mp3"
    open(path, "wb").write(r.content)

    return path