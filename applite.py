import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import threading
import numpy as np
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import whisper
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import io
import time
import random
import gc
import traceback

from logic import analyze_navigation_zones

# --- 1. SETUP & CONFIG ---
@st.cache_resource
def load_models():
    yolo = YOLO('yolov8n.pt')
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    moondream = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    stt_model = whisper.load_model("tiny", device="cpu")
    torch.cuda.empty_cache()
    gc.collect()

    return yolo, moondream, tokenizer, stt_model

try:
    yolo_model, moondream_model, moondream_tokenizer, whisper_model = load_models()
    print(" Models Loaded")
except Exception as e:
    print(f" Model Load Error: {e}")

lock = threading.Lock()
shared_state = {
    "frame": None,
    "navigation_active": False,
    "target_object": "None",
    "frame_count": 0
}

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=50)
    st.title("BlindAssist Lite")
    st.success("⚡ System Stable")

    st.divider()

    st.header(" Vision Modes")
    enable_nav = st.checkbox(" Enable Path Guide", value=False)
    shared_state["navigation_active"] = enable_nav

    st.divider()

    st.header("🔍 Object Finder")
    target = st.selectbox(
        "Highlight Specific Item:",
        ["None", "cell phone", "bottle", "cup", "book", "laptop", "mouse", "person"]
    )
    shared_state["target_object"] = target

# --- 3. VIDEO PROCESSING LOOP ---
def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")

        with lock:
            shared_state["frame_count"] += 1
            frame_num = shared_state["frame_count"]
            shared_state["frame"] = img.copy()
            is_nav_on = shared_state["navigation_active"]
            target_obj = shared_state["target_object"]

        # WARM-UP: Skip first 30 frames to let connection stabilize
        if frame_num < 30:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # AI LOGIC
        should_run_yolo = is_nav_on or (target_obj != "None")

        if not should_run_yolo:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        results = yolo_model(img, verbose=False)

        if results[0].boxes:
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                detections.append(box.xyxy[0].tolist())

                box_color = (0, 255, 0)
                thickness = 2
                draw_box = False

                # Target Match
                if target_obj != "None" and target_obj.lower() == label.lower():
                    box_color = (255, 0, 255)
                    thickness = 4
                    draw_box = True
                    cv2.putText(img, f"FOUND: {label.upper()}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                if draw_box:
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

            if is_nav_on:
                nav_text, nav_color = analyze_navigation_zones(img, detections)
                h, w, _ = img.shape
                cv2.line(img, (int(w*0.33), 0), (int(w*0.33), h), (200,200,200), 1)
                cv2.line(img, (int(w*0.66), 0), (int(w*0.66), h), (200,200,200), 1)
                cv2.putText(img, nav_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, nav_color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    except Exception as e:
        print(f"Callback Error: {e}")
        return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

# --- 4. MAIN UI LAYOUT ---
st.title("👁️ Blind Assist: Stable Demo")

#  Store the random key in session state so it doesn't change on refresh
if "webrtc_key" not in st.session_state:
    st.session_state["webrtc_key"] = f"blind-assist-{random.randint(0, 100000)}"

rtc_config = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
    ]}
)

webrtc_streamer(
    key=st.session_state["webrtc_key"], 
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.divider()

# --- 5. SMART ASSISTANT ---
st.subheader("🤖 AI Assistant")

tab1, tab2, tab3 = st.tabs(["🎤 Voice Mode", "📄 Text Reader", "⚠️ Hazard Check"])

with tab1:
    st.write("Tap Record and ask a question.")
    audio_data = mic_recorder(start_prompt="🎙️ Ask Question", stop_prompt="⏹️ Stop", key='recorder')
    if audio_data:
        with open("temp.wav", "wb") as f: f.write(audio_data['bytes'])
        st.info("Listening...")
        text = whisper_model.transcribe("temp.wav")["text"]
        st.write(f"**You:** {text}")
        with lock: frame = shared_state["frame"]
        if frame is not None:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enc_img = moondream_model.encode_image(pil_img)
            answer = moondream_model.answer_question(enc_img, text, moondream_tokenizer)
            st.success(f"**AI:** {answer}")
            mp3_fp = io.BytesIO()
            tts = gTTS(text=answer, lang='en')
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp, format='audio/mpeg', start_time=0)

with tab2:
    st.write("Use this to read signs or books.")
    if st.button("📖 Read Text in Scene"):
        with lock: frame = shared_state["frame"]
        if frame is not None:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            enc_img = moondream_model.encode_image(pil_img)
            prompt = "Read the text in this image. If no text, say 'No text found'."
            answer = moondream_model.answer_question(enc_img, prompt, moondream_tokenizer)
            st.info(f"**Detected Text:** {answer}")
            mp3_fp = io.BytesIO()
            tts = gTTS(text=f"The text says: {answer}", lang='en')
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp, format='audio/mpeg', start_time=0)

with tab3:
    st.write("Scan for TRAPS with Distance Estimation.")
    if st.button("🚨 Scan for Hazards", type="primary"):
        with lock: frame = shared_state["frame"]
        if frame is not None:
            with st.spinner("Analyzing environment..."):
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enc_img = moondream_model.encode_image(pil_img)
                prompt = (
                    "Describe the main object on the floor. "
                    "Is it a deep hole, a staircase, a wet slippery floor, or construction cones? "
                    "If it is a hazard, say 'UNSAFE: [Name]'. "
                    "If it is safe furniture, say 'SAFE: [Name]'. "
                    "Do not give a one-word answer."
                )
                answer = moondream_model.answer_question(enc_img, prompt, moondream_tokenizer)

            st.warning(f"🤖 Brain Output: '{answer}'")
            ans_lower = answer.lower()
            is_unsafe = ("unsafe" in ans_lower or "stair" in ans_lower or "hole" in ans_lower or "wet" in ans_lower or "cone" in ans_lower or "drop" in ans_lower)
            is_safe_object = "laptop" in ans_lower or "computer" in ans_lower or "table" in ans_lower or "chair" in ans_lower

            if is_unsafe and not is_safe_object:
                st.error(f"⚠️ DANGER: {answer}")
                mp3_fp = io.BytesIO()
                tts = gTTS(text=f"Warning! {answer}", lang='en')
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                st.audio(mp3_fp, format='audio/mpeg', start_time=0)
            else:
                st.success(f"Safe: {answer}")
                mp3_fp = io.BytesIO()
                tts = gTTS(text=f"Safe. {answer}", lang='en')
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                st.audio(mp3_fp, format='audio/mpeg', start_time=0)
