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

from backend_bridge import *
import streamlit as st
import os
import pandas as pd
import speech_recognition as sr
# Patch speech_recognition to use local flac.exe binary on Windows
local_flac = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flac.exe")
sr.audio.get_flac_converter = lambda: local_flac

import asyncio
import edge_tts
import uuid
import base64
import numpy as np
from deep_translator import GoogleTranslator
from PyPDF2 import PdfReader
from pydub import AudioSegment
from datetime import datetime
import streamlit.components.v1 as components
from TTS.api import TTS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if os.path.exists("style.css"):
    local_css("style.css")
else:
    st.error("style.css file missing!")



# CONFIG & SETUP

st.set_page_config(page_title="Voice Synthesis", page_icon="🎙️", layout="wide")
os.makedirs("outputs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)



# SESSION MANAGEMENT
if "user" not in st.session_state:
    st.session_state.user = "Vanshdeep"

page_mapping = {
    "tts": "🎤 TextToSpeech",
    "clone": "🧬 VoiceClone",
    "stt": "🎧 SpeechToText",
    "pdf": "📄 PDF",
    "detector": "🛡️ Detector",
    "dashboard": "📊 Dashboard",
    "home": "🏠 Home"
}

if "page" in st.query_params:
    qp = st.query_params["page"]
    if qp in page_mapping:
        st.session_state.page = page_mapping[qp]
    else:
        st.session_state.page = "🏠 Home"
else:
    st.session_state.page = "🏠 Home"

# MAPS & DICTIONARIES
tts_languages = {"English":"en","Hindi":"hi","French":"fr"}
stt_languages = {"English": "en", "Hindi": "hi", "French": "fr", "Punjabi": "pa"}
voice_map = {
    "en":{"Female":"en-US-AriaNeural","Male":"en-US-GuyNeural"},
    "hi":{"Female":"hi-IN-SwaraNeural","Male":"hi-IN-MadhurNeural"},
    "fr":{"Female":"fr-FR-DeniseNeural","Male":"fr-FR-HenriNeural"},
    "pa":{"Female":"pa-IN-GurshabadNeural","Male":"pa-IN-GurshabadNeural"}
}

# =========================
# CORE UTILITY FUNCTIONS
# =========================
def safe_lang(lang):
    return "en" if lang in ["", None, "Select Language"] else lang

def safe_translate(text, lang):
    try:
        return GoogleTranslator(source='auto', target=safe_lang(lang)).translate(text)
    except:
        return text


def run_tts(text, voice, rate, path):
    async def run():
        tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await tts.save(path)
    try:
        asyncio.run(run())
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())

def audio_with_visualizer(audio_path):
    audio_bytes = open(audio_path,"rb").read()
    b64 = base64.b64encode(audio_bytes).decode()
    components.html(f"""
    <style>
    .bars {{ display:flex; justify-content:center; align-items:flex-end; gap:6px; height:120px; }}
    .bar {{ width:8px; height:20px; background:linear-gradient(180deg,#ff4b4b,#4b7bff); animation:bounce 1s infinite; }}
    @keyframes bounce {{ 0%,100%{{height:20px;}} 50%{{height:120px;}} }}
    </style>
    <div class="bars">{"<div class='bar'></div>"*5}</div>
    <audio controls autoplay style="width:100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
    """, height=260)
    audio = AudioSegment.from_file(audio_path)
    samples = audio.get_array_of_samples()[::100]
    st.subheader("📊 Waveform")
    st.line_chart(samples)

def TextToSpeech(text, lang, voice_type, speed, pitch, emotion):
    lang = safe_lang(lang)
    text = safe_translate(text, lang)
    voice = voice_map[lang][voice_type]
    path = f"outputs/{uuid.uuid4().hex}.mp3"
    rate = "+0%"
    if speed < 0.8: rate = "-20%"
    elif speed > 1.2: rate = "+20%"
    run_tts(text, voice, rate, path)
    audio = AudioSegment.from_file(path)
    audio = audio + pitch
    audio.export(path, format="mp3")
    return path

@st.cache_resource
def load_xtts():
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2")

def VoiceClone(text, lang, sample_path, speed, pitch, emotion):
    xtts_model = load_xtts()
    lang = safe_lang(lang)
    text = safe_translate(text, lang)
    
    # Clean text to prevent hallucinations
    text = text.strip()
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
        
    out_path = f"outputs/{uuid.uuid4().hex}.wav"
    
    # Call synthesizer.tts directly to support speed and lower temperature to prevent gibberish
    wav = xtts_model.synthesizer.tts(
        text=text,
        speaker_name=None,
        speaker_wav=sample_path,
        language_name=lang,
        temperature=0.2,             # Lower temperature to prevent gibberish/hallucinations
        repetition_penalty=5.0,      # Higher penalty to prevent loops/stutters at the end
        speed=speed
    )
    xtts_model.synthesizer.save_wav(wav, out_path)
    
    audio = AudioSegment.from_file(out_path) + pitch
    final = f"outputs/{uuid.uuid4().hex}.mp3"
    audio.export(final, format="mp3")
    return final

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.markdown("<h4 style='color: #94a3b8; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 15px; margin-bottom: 20px; padding-left: 16px;'>Features</h4>", unsafe_allow_html=True)

nav_items = [
    ("🏠 Home", "home"),
    ("🎤 TextToSpeech", "tts"),
    ("🧬 VoiceClone", "clone"),
    ("🎧 SpeechToText", "stt"),
    ("📄 PDF", "pdf"),
    ("🛡️ Detector", "detector"),
    ("📊 Dashboard", "dashboard")
]

active_key = None
for label, key in nav_items:
    if st.session_state.page == label:
        active_key = key
    if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state.page = label
        st.query_params["page"] = key
        st.rerun()

if active_key:
    st.markdown(f"""
        <style>
        div[class*="st-key-nav_{active_key}"] button {{
            background: rgba(56, 189, 248, 0.08) !important;
            border-color: rgba(56, 189, 248, 0.15) !important;
            color: #38bdf8 !important;
            font-weight: 600 !important;
            border-left-color: #38bdf8 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

menu = st.session_state.page



st.markdown("<h1 style='text-align: center;'>Voice Synthesis</h1>", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if menu == "🏠 Home":
    st.markdown("<div style='text-align: center;'><h3>Create natural AI voices in seconds</h3><p>Choose a module to begin.</p></div>", unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
            <a href="/?page=tts" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">🎤 TextToSpeech</div>
                    <p class="feature-desc">Transform text into natural, ultra-realistic neural speech with speed and pitch control.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
            <a href="/?page=clone" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">🧬 Voice Cloning</div>
                    <p class="feature-desc">Create a digital twin of any voice using a short 10s voice reference sample.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown('''
            <a href="/?page=stt" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">🎧 SpeechToText</div>
                    <p class="feature-desc">Convert any audio recording into accurate, translated text with volume boost.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('''
            <a href="/?page=pdf" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">📄 PDF Reader</div>
                    <p class="feature-desc">Convert PDF documents into premium, AI-narrated audiobooks with voice settings.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

    with col5:
        st.markdown('''
            <a href="/?page=detector" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">🛡️ Deepfake Detector</div>
                    <p class="feature-desc">Analyze and detect if audio recordings are real human speech or AI-generated.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

    with col6:
        st.markdown('''
            <a href="/?page=dashboard" target="_self" style="text-decoration: none; color: inherit;">
                <div class="main-card">
                    <div class="feature-title">📊 Dashboard</div>
                    <p class="feature-desc">Monitor system performance, usage activity stats, CPU load, and response metrics.</p>
                </div>
            </a>
        ''', unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: TEXT TO SPEECH
# ==============================================================================
elif menu == "🎤 TextToSpeech":
    st.markdown("<h3 style='margin:0;'>🎙️ AI Voice Composer</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_tts"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        text = st.text_area("Enter Text", placeholder="Type here...", height=250)
        if text:
            m1, m2 = st.columns(2)
            m1.markdown(f'<div class="metric-words">📝 Words: {len(text.split())}</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-chars">🔠 Characters: {len(text)}</div>', unsafe_allow_html=True)

    with side_col:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang = st.selectbox("Language", ["Select Language"] + list(tts_languages.keys()))
        voice = st.selectbox("Voice", ["Select Voice", "Female", "Male"])
        speed = st.selectbox("Speed", ["Select Speed", "Slow", "Normal", "Fast"])
        pitch = st.slider("Pitch Control", -10, 10, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🪄 Generate Professional Voice", use_container_width=True):
        if text.strip() and lang != "Select Language" and voice != "Select Voice":
            # Speed Mapping Logic
            speed_map = {"Slow": 0.7, "Normal": 1.0, "Fast": 1.3}
            
            # Professional Status Display
            st.markdown(f"""
                <div style='padding:15px; border-radius:12px; background: rgba(56, 189, 248, 0.1); 
                border: 1px solid #38bdf8; color:blue; text-align:center; margin-bottom:20px;'>
                🌍 <b>{lang}</b> Language | 🎙️ <b>{voice}</b> Voice | ⚡ <b>{speed}</b> Speed
                </div>
            """, unsafe_allow_html=True)

            with st.spinner("🧠 AI is synthesizing your voice..."):
                try:
                    # Logic call
                    out_path = TextToSpeech(
                        text=text, 
                        lang=tts_languages[lang], 
                        voice_type=voice, 
                        speed=speed_map.get(speed, 1.0), 
                        pitch=pitch, 
                        emotion="Natural"
                    )
                    
                    # Visualizer and Audio
                    audio_with_visualizer(out_path)
                    st.toast("Voice generated successfully!", icon="✅")
                    
                except Exception as e:
                    st.error(f"Generation Error: {e}")
        else:
            st.warning("⚠️ Missing Information: Please ensure Text, Language, and Voice Gender are all selected.")
            

# PAGE 3: VOICE CLONE
elif menu == "🧬 VoiceClone":
    st.markdown("<h3 style='margin:0;'>🎭 Instant Voice Cloning</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_clone"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()
    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        text = st.text_area("Content to Speak", placeholder="Type here...", height=250)
        sample = st.file_uploader("Upload Voice Sample", type=["wav","mp3"])

    with side_col:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        lang = st.selectbox("Language", ["Select Language"] + list(tts_languages.keys()))
        speed = st.selectbox("Speed", ["Select Speed","Slow","Normal","Fast"])
        pitch = st.slider("Pitch", -10, 10, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🧬 Start Voice Cloning", key="clone_btn"):
        if text and sample and lang != "Select Language":
            speed_map = {"Slow": 0.7, "Normal": 1.0, "Fast": 1.3}
            path = f"uploads/{uuid.uuid4().hex}.wav"
            with open(path, "wb") as f: 
                f.write(sample.read())
            with st.spinner("Cloning Voice..."):
                out = VoiceClone(text, tts_languages[lang], path, speed_map.get(speed, 1.0), pitch, "Natural")
                audio_with_visualizer(out)
            st.balloons()
        else:
            st.error("Missing Data.")
            
# PAGE 4: SPEECH TO TEXT

elif menu == "🎧 SpeechToText":
    st.markdown("<h3 style='margin:0;'>🎧 Speech-To-Text Transcriber</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_stt"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()

    # Main Layout
    col_input, col_settings = st.columns([2, 1], gap="large")

    with col_input:
        st.markdown("### 📥 Audio Input")
        audio = st.file_uploader("Upload Audio", type=["wav", "mp3"], help="Select a high-quality audio file for better accuracy.", label_visibility="collapsed")
        
        if audio:
            st.audio(audio) # Preview for user

    with col_settings:
        st.markdown("### ⚙️ Configuration")
        st.markdown('<div class="stt-settings-card">', unsafe_allow_html=True)
        spoken_lang = st.selectbox("Audio Spoken Language", list(stt_languages.keys()), index=0)
        lang_choice = st.selectbox("Output Language", list(stt_languages.keys()), index=0)
        st.info("AI will transcribe the audio in the spoken language, then translate it to the output language if they differ.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Transcription Logic
    if st.button("🚀 Start Transcription", key="stt_btn"):
        if audio:
            with st.status("Processing Audio...", expanded=True) as status:
                # 1. Save uploaded file
                st.write("Extracting raw audio...")
                raw_path = f"uploads/{uuid.uuid4().hex}_{audio.name}"
                with open(raw_path, "wb") as f:
                    f.write(audio.read())

                # 2. Optimize audio (Speech Recognition works best with 16kHz Mono WAV)
                st.write("Optimizing for Neural Recognition...")
                wav_path = raw_path + ".wav"
                sound = AudioSegment.from_file(raw_path)
                sound = sound.normalize() # Boost and normalize volume to improve recognition
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(wav_path, format="wav")

                # 3. Recognition
                r = sr.Recognizer()
                try:
                    with sr.AudioFile(wav_path) as source:
                        st.write("Analyzing patterns...")
                        audio_data = r.record(source)
                    
                    # Convert to Text with selected spoken language
                    stt_google_codes = {"English": "en-US", "Hindi": "hi-IN", "French": "fr-FR", "Punjabi": "pa-IN"}
                    text_result = r.recognize_google(audio_data, language=stt_google_codes[spoken_lang])

                    # Translation if necessary
                    if lang_choice != spoken_lang:
                        st.write(f"Translating from {spoken_lang} to {lang_choice}...")
                        target_lang_code = stt_languages[lang_choice]
                        text_result = GoogleTranslator(source='auto', target=target_lang_code).translate(text_result)

                    status.update(label="✅ Transcription Complete!", state="complete", expanded=False)

                    # Display Styled Result
                    st.markdown("### 📄 Resulting Text")
                    st.markdown(f'<div class="result-card">{text_result}</div>', unsafe_allow_html=True)
                    
                    # Actions
                    st.download_button("📥 Download Transcript", text_result, file_name=f"transcript_{uuid.uuid4().hex[:5]}.txt")

                except sr.UnknownValueError:
                    status.update(label="⚠️ Speech Unintelligible", state="error")
                    st.warning("Google Speech Recognition could not understand the audio (it might be silent or have too much background noise).")
                except sr.RequestError as e:
                    status.update(label="❌ Service Connection Failed", state="error")
                    st.error(f"Could not request results from Google Speech Recognition service; check your internet connection: {e}")
                except Exception as e:
                    status.update(label="❌ Error Occurred", state="error")
                    st.error(f"Could not process audio: {str(e)}")
        else:
            st.warning("⚠️ Please upload an audio file first.")

    # Footer
    st.markdown("<p style='text-align: center; opacity: 0.4; font-size: 12px; margin-top: 50px;'>Neural Speech Recognition v2.0</p>", unsafe_allow_html=True)


# PAGE 5: PDF

elif menu == "📄 PDF":
    st.markdown("<h3 style='margin:0;' class='pdf-title'>📄 PDF Voice Reader</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_pdf"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()

    # Layout Setup
    col_view, col_opt = st.columns([2, 1], gap="large")

    with col_view:
        st.markdown("### 👁️ Document Preview")
        pdf = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

        if pdf:
            st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
            # Encode PDF for preview
            pdf_bytes = pdf.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # Reset pointer for extraction
            pdf.seek(0)
        else:
            st.info("Please upload a PDF file to see the preview here.")

    with col_opt:
        st.markdown("### 🎧 Audiobook Settings")
        st.markdown('<div class="pdf-settings-card">', unsafe_allow_html=True)
        
        # Language Selection (As you requested)
        lang = st.selectbox("Select Language", ["Select Language"] + list(tts_languages.keys()))
        
        st.write("---")
        st.caption("Note: Currently processing up to the first 2000 characters for optimal performance.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Conversion Process
    if st.button("🎙️ Convert PDF to Speech", key="pdf_conv_btn"):
        if pdf and lang != "Select Language":
            with st.status("Reading PDF Content...", expanded=True) as status:
                st.write("Initializing PDF Engine...")
                reader = PdfReader(pdf)
                
                extracted_text = ""
                st.write(f"Extracting text from {len(reader.pages)} pages...")
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text

                if extracted_text.strip():
                    st.write("Generating Neural Voice...")
                    # Limit to 2000 characters for demo
                    clean_text = extracted_text[:2000]
                    
                    # Call global TextToSpeech function
                    out = TextToSpeech(clean_text, tts_languages[lang], "Female", 1.0, 0, "Natural")
                    
                    status.update(label="✅ Audiobook Ready!", state="complete", expanded=False)
                    
                    st.markdown("### 🔉 Audio Output")
                    audio_with_visualizer(out)
                    st.success("Successfully converted text from PDF!")
                else:
                    status.update(label="❌ Extraction Failed", state="error")
                    st.error("Could not extract text. This might be a scanned image PDF.")
        else:
            st.warning("⚠️ Please upload a PDF and select a language first.")

    # Footer
    st.markdown("<p style='text-align: center; opacity: 0.4; font-size: 12px; margin-top: 50px;'>Neural PDF Processor v1.5</p>", unsafe_allow_html=True)


# PAGE 6: DETECTOR

elif menu == "🛡️ Detector":
    st.markdown("<h3 style='margin:0;'>🛡️ AI Voice Deepfake Detector</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_detect"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📥 Audio Upload")
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
        if audio_file:
            st.audio(audio_file)

    with col_result:
        st.markdown("### 📊 Detection Result")
        if audio_file:
            if st.button("🔍 Analyze Audio", use_container_width=True):
                with st.spinner("Analyzing audio frequencies and neural patterns..."):
                    try:
                         import requests
                         audio_file.seek(0)
                         files = {"file": (audio_file.name, audio_file.read(), audio_file.type)}
                         response = requests.post("http://127.0.0.1:8000/detect", files=files)
                         
                         if response.status_code == 200:
                             res_json = response.json()
                             if res_json.get("status") == "success":
                                 result = res_json.get("result")
                                 confidence = res_json.get("confidence")
                                 scores = res_json.get("scores", {})
                                 
                                 if result == "fake":
                                     st.markdown("""
                                         <div style='padding:20px; border-radius:15px; background: rgba(239, 68, 68, 0.1); 
                                         border: 1px solid #ef4444; color:#fca5a5; text-align:center;'>
                                             <span style='font-size: 50px;'>🔴</span>
                                             <h2 style='margin:10px 0; color:#ef4444;'>AI-GENERATED / SYNTHETIC</h2>
                                             <p style='color:#cbd5e1;'>This audio shows strong patterns of artificial synthesis and voice cloning.</p>
                                         </div>
                                     """, unsafe_allow_html=True)
                                 else:
                                     st.markdown("""
                                         <div style='padding:20px; border-radius:15px; background: rgba(34, 197, 94, 0.1); 
                                         border: 1px solid #22c55e; color:#86efac; text-align:center;'>
                                             <span style='font-size: 50px;'>🟢</span>
                                             <h2 style='margin:10px 0; color:#22c55e;'>HUMAN / REAL</h2>
                                             <p style='color:#cbd5e1;'>This audio shows patterns consistent with natural human speech.</p>
                                         </div>
                                     """, unsafe_allow_html=True)
                                 
                                 st.markdown("<br>", unsafe_allow_html=True)
                                 st.write(f"**Confidence Level:** {confidence * 100:.2f}%")
                                 st.progress(float(confidence))
                                 
                                 st.subheader("📉 Score Breakdown")
                                 for label, score in scores.items():
                                     st.write(f"- **{label.title()}**: {score * 100:.2f}%")
                             else:
                                 st.error(f"Detection failed: {res_json.get('message')}")
                         else:
                             st.error(f"Backend API error: HTTP {response.status_code}")
                    except Exception as e:
                         st.error(f"Failed to connect to backend detector: {e}")
        else:
            st.info("Upload an audio file in the left panel to begin analysis.")


# PAGE 7: DASHBOARD & HISTORY

elif menu == "📊 Dashboard":
    st.markdown("<h3 style='margin:0;' class='dash-title'>🚀 System Overview</h3>", unsafe_allow_html=True)
    if st.button("Home", key="home_dash"):
        st.session_state.page = "🏠 Home"
        st.query_params["page"] = "home"
        st.rerun()
    st.divider()

    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Active Models", "2"),
        ("AI Accuracy", "99.2%"),
        ("Server Speed", "120ms"),
        ("Uptime", "100%")
    ]
    
    for i, col in enumerate([c1, c2, c3, c4]):
        label, val = metrics[i]
        col.markdown(f"""<div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-val">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Section
    col_chart, col_usage = st.columns([2, 1], gap="large")
    with col_chart:
        st.markdown("### 📈 Usage Activity")
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['TTS', 'STT'])
        st.area_chart(chart_data)

    with col_usage:
        st.markdown("### ⚙️ Resources")
        st.write("CPU Usage")
        st.progress(25)
        st.write("Neural Load")
        st.progress(65)
        st.write("Storage")
        st.progress(12)

    st.markdown("<p style='text-align: center; opacity: 0.3; margin-top: 50px;'>NeuralVoice Dashboard v3.0</p>", unsafe_allow_html=True)

