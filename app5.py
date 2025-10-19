# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import webbrowser
from collections import deque
import time

st.set_page_config(page_title="Mood + Age + Gender", layout="wide")
st.title("🎭 Mood, Age & Gender Detection (Live Webcam)")


@st.cache_resource
def load_models():
    base_dir = os.path.join(os.getcwd(), "models")

    mood_model = tf.keras.models.load_model(
        os.path.join(base_dir, "fer2013_emotion_model_improved.h5"), compile=False)
    mood_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    faceProto = os.path.join(base_dir, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(base_dir, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(base_dir, "age_deploy.prototxt")
    ageModel = os.path.join(base_dir, "age_net.caffemodel")
    genderProto = os.path.join(base_dir, "gender_deploy.prototxt")
    genderModel = os.path.join(base_dir, "gender_net.caffemodel")

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    return mood_model, mood_classes, faceNet, ageNet, genderNet

mood_model, mood_classes, faceNet, ageNet, genderNet = load_models()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


if "run_mood" not in st.session_state:
    st.session_state.run_mood = False
if "run_age" not in st.session_state:
    st.session_state.run_age = False
if "mood" not in st.session_state:
    st.session_state.mood = None
if "age_group" not in st.session_state:
    st.session_state.age_group = None
if "gender" not in st.session_state:
    st.session_state.gender = None

def getFaceBox(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append([x1, y1, x2, y2])
    return boxes

def map_age_group(age_label):
    if age_label in ['(0-2)', '(4-6)', '(8-12)', '(15-20)']:
        return "Teen"
    elif age_label in ['(25-32)', '(38-43)', '21-24']:
        return "Adult"
    else:
        return "Senior"

def start_mood():
    st.session_state.run_mood = True

def stop_mood():
    st.session_state.run_mood = False

def start_age():
    st.session_state.run_age = True

def stop_age():
    st.session_state.run_age = False

st.subheader("Controls")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Start Mood Detection", on_click=start_mood)
    st.button("Stop Mood Detection", on_click=stop_mood)

with col2:
    st.button("Start Age & Gender Detection", on_click=start_age)
    st.button("Stop Age & Gender Detection", on_click=stop_age)

with col3:
    if st.button("Recommend Music"):
        mood = st.session_state.mood
        age_group = st.session_state.age_group
        if not mood:
            st.warning("Please detect mood first.")
        elif not age_group:
            st.warning("Please detect age group first.")
        else:
            
            spotify_urls = {
                "Happy": {"Teen": "https://open.spotify.com/playlist/0O7Xq0wm8f1cwJ5W0NG01d?si=37d428da9a83475f",
                          "Adult": "https://open.spotify.com/playlist/5nBhUR9oonbMO249D8uZo3?si=b01f7301402d4390",
                          "Senior": "https://open.spotify.com/playlist/1SMBNfiDjNG6yG8BpgkQNY?si=8968c3483733468a"},
                "Sad": {"Teen": "https://open.spotify.com/playlist/5oFeuabHbMrdUmM2uJiK28?si=71994dd7808a4ae0",
                        "Adult": "https://open.spotify.com/playlist/1GZT26rngweto1enpKwnWv?si=658356b641a24093",
                        "Senior": "https://open.spotify.com/playlist/3z66YlKNJxRava5MreQKKO?si=fa3a17320a0b4a4c"},
                "Neutral": {"Teen": "https://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2?si=7011540a99514cf0",
                            "Adult": "https://open.spotify.com/playlist/37i9dQZF1EIfFW8AEFKKOD?si=b1e60983d29b458a",
                            "Senior": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC?si=9c2231297c1f4021"},
                "Angry": {"Teen": "https://open.spotify.com/playlist/37i9dQZF1EIcRK7JMCMZ3M?si=d41d536c93b14031",
                          "Adult": "https://open.spotify.com/playlist/5F6urWvjOhXBGRZTTlvMnj?si=bef5e0a6c56a4ce5",
                          "Senior": "https://open.spotify.com/playlist/4naJtZA87eLo3hrvpHJUYD?si=611383ef194d413a"},
                "Surprise": {"Teen": "https://open.spotify.com/playlist/7oKaFDvg83yxnth26HPdW5?si=f67aab729f6947a6",
                             "Adult": "https://open.spotify.com/playlist/37i9dQZF1EIg65X9FWVODX?si=fa704beabe5f489b",
                             "Senior": "https://open.spotify.com/playlist/7K9yH9ZjGAYvMh3W7yys87?si=a37c53515cd542de"},
                "Disgust": {"Teen": "https://open.spotify.com/playlist/1GXRoQWlxTNQiMNkOe7RqA?si=229691899e5141d0",
                            "Adult": "hhttps://open.spotify.com/playlist/37i9dQZF1DX9qNs32fujYe?si=38d61aa67da0402c",
                            "Senior": "https://open.spotify.com/playlist/27gN69ebwiJRtXEboL12Ih?si=f9e8b5691c39482c"},
                "Fear": {"Teen": "https://open.spotify.com/playlist/3jGTPbP9CluM3u2YMK5Dga?si=ca62112536054698",
                         "Adult": "https://open.spotify.com/playlist/35zEMIYiYH3V39s2j3XqCL?si=454fd32467e14a70",
                         "Senior": "https://open.spotify.com/playlist/37i9dQZF1E8QeXKSGjaldu?si=5fd17e8e15c64b0c"}
            }
            playlist = spotify_urls.get(mood, {}).get(age_group)
            if playlist:
                st.success(f"Opening playlist for {age_group} - {mood}")
                webbrowser.open(playlist)
            else:
                st.warning("No playlist mapping found for that combination.")

st.markdown("---")
left_col, right_col = st.columns([2, 1])
image_display = left_col.empty()
info_box = right_col.empty()

def run_mood_loop():
    cap = cv2.VideoCapture(0)
    mood_queue = deque(maxlen=10)
    last_display = None
    try:
        while st.session_state.run_mood:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            boxes = getFaceBox(faceNet, frame, conf_threshold=0.6)
            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48)) / 255.0
                gray = np.expand_dims(gray, axis=(0, -1))
                preds = mood_model.predict(gray, verbose=0)[0]
                label = mood_classes[np.argmax(preds)]
                mood_queue.append(label)
                stable = max(set(mood_queue), key=mood_queue.count)
                st.session_state.mood = stable
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, stable, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # update UI once per loop
            image_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            info_box.markdown(f"**Current stored mood:** {st.session_state.mood or 'Not set'}")
            # small sleep to yield control so Stop button press is registered quickly
            time.sleep(0.05)
    finally:
        cap.release()

def run_age_loop():
    cap = cv2.VideoCapture(0)
    age_queue = deque(maxlen=10)
    try:
        while st.session_state.run_age:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            boxes = getFaceBox(faceNet, frame, conf_threshold=0.6)
            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                gpred = genderNet.forward()[0]
                gender = genderList[np.argmax(gpred)]

                ageNet.setInput(blob)
                apred = ageNet.forward()[0]
                sorted_idx = np.argsort(apred)[::-1]
                top_bucket = ageList[sorted_idx[0]]
                second_bucket = ageList[sorted_idx[1]]
                if top_bucket == '(15-20)' and apred[sorted_idx[0]] < 0.6 and second_bucket == '(25-32)':
                    age_label = '21-24'
                else:
                    age_label = top_bucket

                age_queue.append(age_label)
                stable_age_label = max(set(age_queue), key=age_queue.count)
                st.session_state.gender = gender
                st.session_state.age_group = map_age_group(stable_age_label)

                label = f"{gender}, {stable_age_label} ({st.session_state.age_group})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            image_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            info_box.markdown(f"**Stored age group:** {st.session_state.age_group or 'Not set'}  \n**Stored gender:** {st.session_state.gender or 'Not set'}")
            time.sleep(0.05)
    finally:
        cap.release()


# When user clicked Start Mood Detection, run the loop (this blocks until stop sets flag False)
if st.session_state.run_mood and not st.session_state.run_age:
    run_mood_loop()

# When user clicked Start Age & Gender Detection, run the loop
if st.session_state.run_age and not st.session_state.run_mood:
    run_age_loop()


st.markdown("---")
st.subheader("Stored Predictions")
st.write(f"- Mood: **{st.session_state.mood or 'Not predicted'}**")
st.write(f"- Age group: **{st.session_state.age_group or 'Not predicted'}**")
st.write(f"- Gender: **{st.session_state.gender or 'Not predicted'}**")
