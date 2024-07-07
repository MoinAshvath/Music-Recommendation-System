import streamlit as st
import json
import time
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from streamlit_lottie import st_lottie

# @st.cache_data
# def load_pickled_data():                                        # Cached to load faster
#     with open("recommend_songs_by_name.pkl", "rb") as f:
#         recommend_songs_by_name, dataset, Mtr, Mtr1 = pickle.load(f)
#     return recommend_songs_by_name, dataset, Mtr, Mtr1
# recommend_songs_by_name, dataset, Mtr, Mtr1 = load_pickled_data()

# @st.cache_data
# def load_lottiefile(animation: str):
#     with open(animation, "r") as f:
#         return json.load(f)
# lottie_file = load_lottiefile("animation/ani2.json")


# Load the pickled function and dataset
with open("recommend_songs_by_name.pkl", "rb") as f:
    recommend_songs_by_name,dataset,Mtr,Mtr1 = pickle.load(f)

def load_lottiefile(animation: str):
    with open(animation,"r") as f:
        return json.load(f)
    
lottie_file = load_lottiefile("animation/ani2.json")


def get_album_cover_path(song_name):
    album_cover_folder = "images"
    file_name = f"{song_name}.jpg"
    file_path = os.path.join(album_cover_folder, file_name)
    
    if os.path.exists(file_path):
        return file_path
    else:
        return os.path.join(album_cover_folder, "spotify.jpg")



# Creating a dictionary to map track names to indices
song_name_to_index = {str(song_name).lower(): index for index, song_name in enumerate(Mtr)}
song_name_to_index = pd.Series(song_name_to_index)


def recommend_songs_by_name(input_song_name, k_value, dataset):
    input_song_name = str(input_song_name).lower()             # Checking if the input song name exists in the mapping dictionary

    if input_song_name not in song_name_to_index:
        return "Song not found in the dataset"
    input_song_index = song_name_to_index[input_song_name]     # Getting the index of the input song

    knn_model = NearestNeighbors(n_neighbors=k_value, algorithm='auto')       # Initializing the NearestNeighbors model
    knn_model.fit(dataset[['danceability','energy','acousticness','instrumentalness','tempo','duration_ms']])
    # Finding K-nearest neighbors of input song
    _, indices = knn_model.kneighbors([dataset.iloc[input_song_index][['danceability','energy','acousticness','instrumentalness','tempo','duration_ms']]])
    recommended_song_ids = dataset.iloc[indices[0]]['track_id']                   # Get the track IDs of the recommended songs
    return recommended_song_ids            





st.title("Music Recommendation System")

selection = st.sidebar.selectbox("Choose an option: ",["Song-based Recommend", "Emotion-based Recommend"])

if selection=="Song-based Recommend":
    st_lottie(lottie_file,height=150,width=1280,speed=1)
    # User input for the song name
    input_song_name = st.text_input("Enter the song name:")

    # User input for the number of recommendations (k)
    k_value = st.number_input("Number of Recommendations:", min_value=0, max_value=100, step=1)

    if st.button("Get Recommendations"):
        if input_song_name:
            if k_value>0 and k_value<100:
                # Call the recommend_songs_by_name function with the dataset
                recommendations = recommend_songs_by_name(input_song_name, k_value, dataset)
                if isinstance(recommendations, str):
                    st.error(recommendations)  # Song not found in the dataset
                else:
                    #st.success(f"Recommended Songs for '{input_song_name}':")
                    for i, song_id in enumerate(recommendations):
                        st.write(Mtr.iloc[song_id])
                        st.image(get_album_cover_path(Mtr.iloc[song_id]),width=200)
                        st.write()
            else:
                st.warning("Enter a valid number of recommendation")
        else:
            st.warning("Please enter a song name")


if selection=="Emotion-based Recommend":
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer
    import av
    import cv2 
    import numpy as np 
    import mediapipe as mp 
    from keras.models import load_model
    import webbrowser

    model  = load_model("model.h5")
    label = np.load("labels.npy")
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    st.header("Emotion Based Music Recommender")

    if "run" not in st.session_state:
        st.session_state["run"] = "true"

    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion=""

    if not(emotion):
        st.session_state["run"] = "true"
    else:
        st.session_state["run"] = "false"

    class EmotionProcessor:
        def recv(self, frame):
            frm = frame.to_ndarray(format="bgr24")

            ##############################
            frm = cv2.flip(frm, 1)

            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for i in range(42):
                        lst.append(0.0)

                lst = np.array(lst).reshape(1,-1)

                pred = label[np.argmax(model.predict(lst))]

                print(pred)
                cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

                np.save("emotion.npy", np.array([pred]))

                
            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


            ##############################

            return av.VideoFrame.from_ndarray(frm, format="bgr24")

    lang = st.text_input("Language")
    singer = st.text_input("singer")

    if lang and singer and st.session_state["run"] != "false":
        webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

    btn = st.button("Recommend me songs")

    if btn:
        if not(emotion):
            st.warning("Please let me capture your emotion first")
            st.session_state["run"] = "true"
        else:
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
            np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"