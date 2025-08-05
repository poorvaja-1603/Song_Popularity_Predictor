import streamlit as st
import matplotlib.pyplot as plt
import pickle

st.title("Get Prediction of Popularity of any song!")
st.sidebar.image('https://images.squarespace-cdn.com/content/v1/5af1298bfcf7fd60f31f66bd/1628238820419-V7L3V9YCE8MY6EI5ZVL5/spotify+opinion+article-01.png')
st.markdown(
    """
    **Don't know your song's features?**  
    Get them from the [Spotify Developer Console](https://developer.spotify.com/console/get-audio-features-track/):
    1. Go to the above link  
    2. Log in and click the green **"Get Token"** button  
    3. Paste your song's Track ID 
    4. Click **"Try It"** and scroll down to see the song's features  
    5. You will get the required values!
    """
)
with open('model.pkl', 'rb') as f:
    grid = pickle.load(f)
with open('genre_map.pkl', 'rb') as f:
    genre_freq = pickle.load(f)

features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness','liveness', 'valence', 'tempo', 'track_genre_freq', 'duration_s',
              'dance_valence', 'energy_loudness', 'speech_liveness']

importances = grid.best_estimator_.feature_importances_

def get_input():
    dance = st.sidebar.number_input("Enter the danceability of the song : ", min_value=0.0, 
                                    max_value=1.0,  step = 0.0001, value = 0.566)
    energy = st.sidebar.number_input("Enter the energy of the song : ", min_value=0.0, 
                                    max_value=1.0,  step = 0.0001, value = 0.646)
    loudness = st.sidebar.number_input("Enter the loudness of the song : ", min_value=-49.531, 
                                    max_value=4.532,  step = 0.0001, value = -8.253)
    speechiness = st.sidebar.number_input("Enter the speechiness of the song : ", min_value=0.0001, 
                                    max_value=0.965,  step = 0.0001, value = 0.086)
    acoustics = st.sidebar.number_input("Enter the acousticness of the song : ", min_value=0.0001, 
                                    max_value=0.996,  step = 0.0001, value = 0.310)
    instrumental = st.sidebar.number_input("Enter the instrumentalness of the song : ", min_value=0.000, 
                                    max_value=1.000,  step = 0.0001, value = 0.164)
    liveness = st.sidebar.number_input("Enter the liveness of the song : ", min_value=0.0001, 
                                    max_value=1.000,  step = 0.0001, value = 0.218)
    valence = st.sidebar.number_input("Enter the valence of the song : ", min_value=0.0001, 
                                    max_value=0.995,  step = 0.0001, value = 0.469)
    tempo = st.sidebar.number_input("Enter the tempo of the song : ", min_value=0.0001, 
                                    max_value=243.372,  step = 0.0001, value = 122.700)
    genre = st.sidebar.selectbox("Select the genre of the song : ", list(genre_freq.keys()))
    duration = st.sidebar.number_input("Enter the duration of the song in seconds : ", min_value=24.266, 
                                    max_value=5237.295,  step = 0.0001, value = 230.419)
    dan_val = dance * valence
    ener_loud = energy * loudness
    speech_live = speechiness * liveness
    track_genre_freq = genre_freq[genre]

    user_data = [[dance, energy, loudness, speechiness, acoustics, instrumental, liveness, 
                  valence, tempo, track_genre_freq, duration, dan_val,ener_loud, speech_live]]
    return user_data

user_data = get_input()

if st.button("Predict Popularity!"):
    prediction = grid.predict(user_data)
    st.success(f"The popularity score of the song out of 100 is : {prediction}")
    st.info("This model has Mean Absolute Error of 10.12 and Root Mean Squared Error of 14.08")
    st.subheader("Feature Importances")
    fig, ax = plt.subplots()
    ax.barh(features, importances, color = 'lightblue')
    ax.set_xlabel("Importances")
    ax.set_ylabel("Features")
    ax.invert_yaxis()
    st.pyplot(fig)