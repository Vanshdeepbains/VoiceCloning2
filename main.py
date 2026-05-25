import os
import static_ffmpeg

# Setup static ffmpeg and ffprobe in PATH for pydub and other audio libraries
try:
    static_ffmpeg.add_paths()
except Exception as e:
    print(f"Warning: Failed to setup static-ffmpeg paths: {e}")

import torch
import torchaudio
import soundfile as sf
import io
import numpy as np

# Patch torch.load to default to weights_only=False for PyTorch 2.6+ compatibility with Coqui TTS
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load

# Patch torchaudio.load and torchaudio.save to bypass broken torchcodec / FFmpeg dependencies
def patched_torchaudio_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096, backend=None):
    try:
        start = frame_offset
        stop = None if num_frames == -1 else start + num_frames
        data, samplerate = sf.read(uri, start=start, stop=stop, dtype='float32')
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif channels_first:
            tensor = tensor.transpose(0, 1)
        return tensor, samplerate
    except Exception as e:
        try:
            if isinstance(uri, (str, os.PathLike)):
                audio = AudioSegment.from_file(uri)
            else:
                uri.seek(0)
                audio = AudioSegment.from_file(io.BytesIO(uri.read()))
            samplerate = audio.frame_rate
            channel_sounds = audio.split_to_mono()
            samples = [np.array(c.get_array_of_samples(), dtype=np.float32) for c in channel_sounds]
            max_val = 2 ** (8 * audio.sample_width - 1)
            samples = [s / max_val for s in samples]
            data = np.stack(samples, axis=0)
            if frame_offset > 0 or num_frames != -1:
                start = frame_offset
                end = None if num_frames == -1 else start + num_frames
                data = data[:, start:end]
            tensor = torch.from_numpy(data)
            if not channels_first:
                tensor = tensor.transpose(0, 1)
            return tensor, samplerate
        except Exception as ex:
            raise RuntimeError(f"Patched torchaudio.load failed to load audio. Soundfile error: {e}. Pydub error: {ex}")

def patched_torchaudio_save(uri, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None, buffer_size=4096, backend=None, compression=None):
    data = src.detach().cpu().numpy()
    if data.ndim == 1:
        pass
    else:
        if channels_first:
            data = data.T
    sf.write(uri, data, sample_rate)

torchaudio.load = patched_torchaudio_load
torchaudio.save = patched_torchaudio_save

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from pydub import AudioSegment
import edge_tts
import uuid
import os
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)





xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")



# TTS API

@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):

    tts = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    out = f"outputs/{uuid.uuid4().hex}.mp3"
    await tts.save(out)

    return FileResponse(out, media_type="audio/mp3")



# VOICE CLONE

@app.post("/clone")
async def clone(text: str = Form(...), file: UploadFile = File(...), lang: str = Form("en"), speed: float = Form(1.0)):

    in_path = f"uploads/{uuid.uuid4().hex}.wav"
    with open(in_path, "wb") as f:
        f.write(await file.read())

    out_wav = f"outputs/{uuid.uuid4().hex}.wav"

    # Clean text to prevent hallucinations
    cleaned_text = text.strip()
    if cleaned_text and cleaned_text[-1] not in ['.', '!', '?']:
        cleaned_text += '.'

    # Call synthesizer.tts directly to support speed and lower temperature to prevent gibberish
    wav = xtts.synthesizer.tts(
        text=cleaned_text,
        speaker_name=None,
        speaker_wav=in_path,
        language_name=lang,
        temperature=0.2,             # Lower temperature to prevent gibberish/hallucinations
        repetition_penalty=5.0,      # Higher penalty to prevent loops/stutters at the end
        speed=speed
    )
    xtts.synthesizer.save_wav(wav, out_wav)

    final = f"outputs/{uuid.uuid4().hex}.mp3"
    AudioSegment.from_file(out_wav).export(final, format="mp3")

    return FileResponse(final, media_type="audio/mp3")



# STT

@app.post("/stt")
async def stt(file: UploadFile = File(...)):

    path = f"uploads/{uuid.uuid4().hex}.wav"
    with open(path, "wb") as f:
        f.write(await file.read())

    return {"text": "Speech converted (add model here)"}



# PDF

@app.post("/pdf")
async def pdf(file: UploadFile = File(...)):

    from PyPDF2 import PdfReader
    reader = PdfReader(file.file)

    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""

    return {"text": text[:3000]}


# DETECT API

detector_pipeline = None

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global detector_pipeline
    if detector_pipeline is None:
        from transformers import pipeline
        detector_pipeline = pipeline("audio-classification", model="garystafford/wav2vec2-deepfake-voice-detector")
        
    in_path = f"uploads/{uuid.uuid4().hex}_{file.filename}"
    with open(in_path, "wb") as f:
        f.write(await file.read())
        
    try:
        results = detector_pipeline(in_path)
        scores = {item['label']: item['score'] for item in results}
        winner = results[0]['label']
        confidence = results[0]['score']
        
        return {
            "status": "success",
            "result": winner,
            "confidence": confidence,
            "scores": scores
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


@app.get("/")
def root():
    return {"message": "Voice Synthesis Backend Running"}
# uvicorn main:app --re load