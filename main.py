import speech_recognition as sr
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import webbrowser
from datetime import datetime
import json
import time
import spacy
import pickle


import numpy as np
# Uses a custom trained Hugging Face Transformer model

#Stuff to do:
#actually make play music? -> can use spotify? -> use the genres, song name or playlist,  pause, volume-up/down
#i named it gonkfield but it doesnt recognise it as a word, so either change name or make it recognise
#make the timer work, no clue how, probably threads?
#open more apps
#get date
#weather
#calculate
#search google
#open website
#tell joke/fact

# --- CONFIGURATION ---

WAKE_WORD = "hey"

# --- INITIALIZATION ---
recognizer = sr.Recognizer()#speech recognition
tts = pyttsx3.init()#text-to-speech
nlp = spacy.load("gonkfield_ner_model")#loads trained spaCy model
# Load model and tokenizer for intents
tokenizer = AutoTokenizer.from_pretrained("./intent_model")
intent_model = AutoModelForSequenceClassification.from_pretrained("./intent_model")

# Load label encoder
with open("./intent_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

#text-to-speech
def speak(text):
    print(f"Assistant: {text}")
    tts.say(text)
    tts.runAndWait()

#listens to microphone
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("Speech error:", e)
        return ""

#waits for wake word
def wait_for_wake_word():
    while True:
        text = listen()
        if WAKE_WORD in text:
            speak("Yes?")
            return


def extract_entities(text):
    print("extracting entities")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def predict_intent(text):
    print("predicting intent")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = intent_model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    pred_id = np.argmax(logits, axis=1)[0]
    return label_encoder.inverse_transform([pred_id])[0]


def execute_intent(intent, entities):
    if intent == "get_time":
        speak(f"The current time is {datetime.now().strftime('%I:%M %p')}")
        return True

    elif intent == "play_music":
        genre = next((e[0] for e in entities if e[1] == "GENRE"), None) #loops through and finds the first entity that contains the label 'GENRE' and then outputs the text
        if genre:
            speak(f"Playing {genre} music.")
        else:
            speak("Playing some music.")
        return True

    elif intent == "set_timer":
        duration = next((e[0] for e in entities if e[1] == "DURATION"), None)
        if duration:
            speak(f"Setting a timer for {duration}.")
        else:
            speak("For how long?")
        return True

    elif intent == "open_app":
        app = next((e[0] for e in entities if e[1] == "APP"), None)
        if app=="youtube":
            speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com")
        if app=="spotify":
            speak("Opening Spotify.")
            webbrowser.open("https://www.spotify.com")
        return True

    elif intent == "exit":
        speak("Goodbye!")
        return False

    else:
        speak("Sorry, I didn't understand that.")
    return True


# === MAIN LOOP ===

if __name__ == "__main__":
    speak("Voice assistant is online.")
    wait_for_wake_word()
    while True:
        command = listen()
        if command:
            print(f"[DEBUG] Command Received: {command}")
            intent = predict_intent(command)
            entities = extract_entities(command)
            print("[DEBUG] Intent:", intent)
            print("[DEBUG] Entities:", entities)
            if not execute_intent(intent, entities):
                break