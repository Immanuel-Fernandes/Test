import streamlit as st
import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        st.info("Listening for speech...")
        audio = recognizer.listen(source)

    try:
        st.info("Recognizing...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "Could not request results; check your network connection."
    except sr.UnknownValueError:
        return "Could not understand the audio."

st.title("Speech to Text Conversion")

if st.button("Start Recording"):
    result = speech_to_text()
    st.write("You said:")
    st.write(result)
