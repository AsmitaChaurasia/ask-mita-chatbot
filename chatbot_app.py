# ✅ Ask-Mita (Fixed Version): Gemini with Timeout, Spinner, and Cleaner Flow

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import PyPDF2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue
import numpy as np
import scipy.io.wavfile
import speech_recognition as sr
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# ✅ Load environment variables
load_dotenv()

# ✅ Gemini API setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Load Dataset
DATASET_PATH = "chatbot_dataset.csv"
try:
    if os.path.exists(DATASET_PATH) and os.stat(DATASET_PATH).st_size > 0:
        df = pd.read_csv(DATASET_PATH, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.DataFrame(columns=["question", "answer"])
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    df = pd.DataFrame(columns=["question", "answer"])

# ✅ Ask Gemini with timeout and spinner
def ask_gemini_safe(prompt, timeout=30):
    try:
        with st.spinner("🤖 Thinking..."):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(ask_gemini_raw, prompt)
                return future.result(timeout=timeout)
    except FuturesTimeout:
        return "⚠️ Gemini API took too long to respond. Try again."
    except Exception as e:
        return f"❌ Gemini error: {e}"

def ask_gemini_raw(prompt):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text

# ✅ Streamlit setup
st.set_page_config(page_title="Ask-Mita", page_icon="🤖")
st.title("🤖 Ask-Mita")
st.markdown("Ask me anything!")

# ✅ Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Audio setup
class AudioProcessor:
    def __init__(self): self.q = queue.Queue(maxsize=100)
    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten().astype("int16")
        try: self.q.put_nowait(audio)
        except queue.Full: pass
        return frame
    def get_audio_data(self):
        data = []
        while not self.q.empty():
            data.extend(self.q.get())
        return data

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor
)

voice_input = ""
if ctx.audio_processor and st.button("🎧 Transcribe Audio"):
    audio_data = ctx.audio_processor.get_audio_data()
    ctx.audio_processor.q.queue.clear()
    if audio_data:
        temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
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
                st.error(f"Speech Recognition error: {e}")

# ✅ PDF Upload (optional)
uploaded_file = st.file_uploader("📄 Upload a PDF to ask questions about it", type="pdf")
pdf_context = ""
if uploaded_file:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        pdf_context = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if len(pdf_context.split()) > 1000:
            pdf_context = " ".join(pdf_context.split()[:1000])  # Limit context size
    except:
        st.warning("❗ Failed to read PDF")

# ✅ Input
typed_input = st.text_input("Type your question here:")
user_input = voice_input or typed_input

if st.button("Submit") and user_input.strip():
    final_answer = ""

    # 1️⃣ Check local CSV dataset
    matched = df[df['question'].str.lower() == user_input.lower()]
    if not matched.empty:
        final_answer = matched.iloc[0]['answer']
    else:
        prompt = f"Context: {pdf_context}\n\nQuestion: {user_input}" if pdf_context else user_input
        final_answer = ask_gemini_safe(prompt)

    # ✅ Save chat
    st.session_state.chat_history.append({"question": user_input, "answer": final_answer})

    # ✅ Show answer
    st.markdown("---")
    st.subheader("🤖 Mita's Answer")
    st.write(final_answer)

    # ✅ Feedback
    feedback = st.radio("Was this helpful?", ["Yes", "No"], key=user_input)
    if feedback:
        with open("feedback.csv", "a", encoding="utf-8") as f:
            f.write(f'"{user_input}","{final_answer}","{feedback}"\n')
        st.success("✅ Feedback saved")

# ✅ Show chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"🧑‍💻 **You**: {chat['question']}")
        st.markdown(f"🤖 **Mita**: {chat['answer']}")

# ✅ Export option
if st.sidebar.button("📄 Export Chat to CSV"):
    pd.DataFrame(st.session_state.chat_history).to_csv("chat_export.csv", index=False)
    st.sidebar.success("Exported to chat_export.csv")
