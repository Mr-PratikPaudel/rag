
"""pipenv install langchain-huggingface langchain-text-splitters pypdf langchain-community sentence-transformers faiss-cpu"""

import streamlit as st
import os
import hashlib
import asyncio
import edge_tts
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
st.set_page_config(page_title="Medical Voice RAG", page_icon="🩺", layout="wide")
st.title("🩺 Medical Voice RAG Assistant")
st.caption("Ask multiple medical questions using your voice. Answers are generated only from the medical PDF.")
st.warning("⚠️ This application is a document-based medical assistant. It does not replace a qualified medical professional.")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
@st.cache_resource
def get_vectorstore():
    pdf_name = "./medical.pdf"
    if not os.path.exists(pdf_name):
        raise FileNotFoundError(f"Medical PDF not found: {pdf_name}")
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
def transcribe_audio(audio_file):
    transcription = groq_client.audio.transcriptions.create(file=("voice.wav", audio_file.getvalue()), model="whisper-large-v3", response_format="json", temperature=0.0)
    return transcription.text
def generate_voice(text):
    audio_file = "medical_answer.mp3"
    voice = "zh-HK-HiuMaanNeural"
    communicate = edge_tts.Communicate(text=text, voice=voice)
    asyncio.run(communicate.save(audio_file))
    with open(audio_file, "rb") as audio:
        audio_bytes = audio.read()
    os.remove(audio_file)
    return audio_bytes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.subheader("🎙️ Tell me your medical problem")
audio_value = st.audio_input("Record a new medical problem", sample_rate=16000)
if audio_value:
    audio_bytes = audio_value.getvalue()
    current_audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if current_audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = current_audio_hash
        try:
            with st.spinner("🎧 Listening and transcribing..."):
                user_text = transcribe_audio(audio_value)
            if not user_text.strip():
                st.error("I could not understand the audio.")
                st.stop()
            st.subheader("📝 Transcribed Problem")
            st.info(user_text)
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.spinner("🔎 Searching the medical document..."):
                vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            groq_chat = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model="llama-3.1-8b-instant", temperature=0)
            medical_prompt = ChatPromptTemplate.from_template("""
You are a document-grounded medical information assistant.
STRICT RULES:
1. Answer ONLY using the CONTEXT.
2. The CONTEXT comes from the medical PDF.
3. Do NOT use your general medical knowledge.
4. Do NOT use the internet.
5. Do NOT diagnose the patient.
6. Do NOT invent symptoms, diseases, medicines, or treatments.
7. Do NOT make assumptions.
8. If the answer is not clearly available in the CONTEXT, respond EXACTLY:
"I could not find the answer in the medical document 再试一次 (zài shì yī cì)."
9. If the CONTEXT contains relevant information, explain it clearly and simply.
10. Give medical information ONLY if explicitly supported by the CONTEXT.
CONTEXT:
{context}
PATIENT PROBLEM:
{question}
Answer only from the medical document.
Start the answer directly.
""")
            with st.spinner("🔎 Finding relevant medical information..."):
                retrieved_docs = retriever.invoke(user_text)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chain = medical_prompt | groq_chat | StrOutputParser()
            with st.spinner("🧠 Reading the medical document..."):
                response = chain.invoke({"context": context, "question": user_text})
            st.subheader("🩺 Medical Document Answer")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.spinner("🔊 Reading the complete answer..."):
                audio_bytes = generate_voice(response)
            st.subheader("🔊 Listen to Complete Answer")
            st.audio(audio_bytes, format="audio/mp3")
            st.success("✅ Complete answer generated. You can record another question now.")
        except Exception as e:
            st.error(f"Application Error: {str(e)}")