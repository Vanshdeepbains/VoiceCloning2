import shutil
import os

def clone_voice(text, speaker_wav):
    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/cloned.wav"

    # Temporary simple demo copy
    shutil.copy(speaker_wav, output_path)

    return output_path
