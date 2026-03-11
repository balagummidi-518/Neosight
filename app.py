import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate',150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech recognition
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return text.lower()
    except:
        return ""

st.title("Neosight - AI Vision for Blind People")

# Language selection
speak("Please say your language. Telugu say one. English say two. Hindi say three.")

st.write("Say 1 Telugu, 2 English, 3 Hindi")

lang_input = listen()

language="english"

if "one" in lang_input or "1" in lang_input:
    language="telugu"
    speak("నియోసైట్ కి స్వాగతం. స్టార్ట్ అని చెప్పండి.")

elif "three" in lang_input or "3" in lang_input:
    language="hindi"
    speak("नियोसाइट में आपका स्वागत है। शुरू करने के लिए स्टार्ट बोलिए।")

else:
    language="english"
    speak("Welcome to Neosight. Say start to begin.")

# Wait for start command
cmd = listen()

if "start" in cmd:

    speak("Starting camera")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    frame_window = st.image([])

    last_voice=""

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        objects=[]

        for r in results:
            for box in r.boxes:

                cls=int(box.cls[0])
                label=r.names[cls]

                objects.append(label)

                x1,y1,x2,y2=map(int,box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

        if objects:

            speech="I see "+", ".join(objects)

            if speech!=last_voice:
                speak(speech)
                last_voice=speech

        frame_window.image(frame,channels="BGR")

    cap.release()