import pyttsx3
import os

def generate_voice(text):
    engine = pyttsx3.init()

    os.makedirs("outputs", exist_ok=True)

    path = "outputs/voice.wav"

    engine.save_to_file(text, path)
    engine.runAndWait()

    return path