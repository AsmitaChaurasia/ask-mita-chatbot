import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import tempfile
import queue
import av
import numpy as np
import scipy.io.wavfile
import speech_recognition as sr
import PyPDF2

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --------------------------------------------------
# 🔐 Load Environment Variables
# --------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------------------------------
# 📊 Load Local Dataset
# --------------------------------------------------
DATASET_PATH = "chatbot_dataset.csv"

try:
    if os.path.exists(DATASET_PATH) and os.stat(DATASET_PATH).st_size > 0:
        df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.DataFrame(columns=["question", "answer"])
except Exception as e:
    st.error(f"Dataset load error: {e}")
    df = pd.DataFrame(columns=["question", "answer"])

# --------------------------------------------------
# 🤖 Gemini Functions (SAFE)
# --------------------------------------------------
def ask_gemini_raw(prompt):
    model = genai.GenerativeModel("models/gemini-1.0-pro")
    response = model.generate_content(prompt)
    return response.text


def ask_gemini_safe(prompt, timeout=30):
    try:
        with st.spinner("🤖 Thinking..."):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(ask_gemini_raw, prompt)
                return future.result(timeout=timeout)
    except FuturesTimeout:
        return "⚠️ Gemini is taking too long. Please try again."
    except Exception:
        return "❌ AI service temporarily unavailable."

# --------------------------------------------------
# 🎧 Audio Processor (Speech Input)
# --------------------------------------------------
class AudioProcessor:
    def __init__(self):
        self.q = queue.Queue(maxsize=100)

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten().astype("int16")
        try:
            self.q.put_nowait(audio)
        except queue.Full:
            pass
        return frame

    def get_audio(self):
        data = []
        while not self.q.empty():
            data.extend(self.q.get())
        return data

# --------------------------------------------------
# 🖥 Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Ask-Mita", page_icon="🤖")
st.title("🤖 Ask-Mita")
st.markdown("Your AI Assistant with Local + Gemini Intelligence")

# --------------------------------------------------
# 💬 Session State
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# 🎧 Audio Input
# --------------------------------------------------
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

voice_input = ""

if ctx.audio_processor and st.button("🎙 Transcribe Audio"):
    audio_data = ctx.audio_processor.get_audio()
    ctx.audio_processor.q.queue.clear()

    if audio_data:
        temp_path = os.path.join(tempfile.gettempdir(), "audio.wav")
        scipy.io.wavfile.write(temp_path, 48000, np.array(audio_data))

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            try:
                voice_input = recognizer.recognize_google(audio)
                st.success(f"You said: {voice_input}")
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError as e:
                st.error(f"Speech error: {e}")

# --------------------------------------------------
# 📄 PDF Upload (Optional Context)
# --------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF (optional)", type="pdf")
pdf_context = ""

if uploaded_file:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        pdf_context = " ".join(
            page.extract_text()
            for page in reader.pages
            if page.extract_text()
        )
        pdf_context = " ".join(pdf_context.split()[:1000])
    except:
        st.warning("PDF could not be read")

# --------------------------------------------------
# ❓ User Input
# --------------------------------------------------
typed_input = st.text_input("Type your question:")
user_input = voice_input or typed_input

# --------------------------------------------------
# 🚀 Submit
# --------------------------------------------------
if st.button("Submit") and user_input.strip():
    answer = ""

    # 🔹 Local dataset first
    matched = df[df["question"].str.lower().str.contains(user_input.lower(), na=False)]

    if not matched.empty:
        answer = matched.iloc[0]["answer"]
    else:
        prompt = f"Context:\n{pdf_context}\n\nQuestion:\n{user_input}" if pdf_context else user_input
        answer = ask_gemini_safe(prompt)

    st.session_state.chat_history.append(
        {"question": user_input, "answer": answer}
    )

    st.markdown("---")
    st.subheader("🤖 Mita says:")
    st.write(answer)

# --------------------------------------------------
# 💬 Chat History
# --------------------------------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")

    for chat in st.session_state.chat_history:
        st.markdown(f"🧑 **You:** {chat['question']}")
        st.markdown(f"🤖 **Mita:** {chat['answer']}")

# --------------------------------------------------
# 📤 Export Chat
# --------------------------------------------------
if st.sidebar.button("📄 Export Chat to CSV"):
    pd.DataFrame(st.session_state.chat_history).to_csv(
        "chat_export.csv", index=False
    )
    st.sidebar.success("Chat exported successfully")
