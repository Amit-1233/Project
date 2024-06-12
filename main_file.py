import streamlit as st
import mysql.connector
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm




def create_connection():
    db_config = st.secrets["connections"]["mysql"]
    conn = mysql.connector.connect(
        host=db_config["host"],
        user=db_config["username"],
        password=db_config["password"],
        database=db_config["database"],
        port=db_config["port"]
    )
    return conn


# Save uploaded file and its features to the database
def save_uploaded_file(uploaded_file, user_id, model):
    try:
        # Save the uploaded file to a directory
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Connect to database
        conn = create_connection()
        cursor = conn.cursor()

        # Check existing entries and delete the oldest if more than 1 entry exists
        cursor.execute("SELECT COUNT(*) FROM user_images WHERE user_id = %s", (user_id,))
        count = cursor.fetchone()[0]
        if count >= 2:
            cursor.execute("SELECT MIN(id) FROM user_images WHERE user_id = %s", (user_id,))
            oldest_id = cursor.fetchone()[0]
            cursor.execute("DELETE FROM user_images WHERE id = %s", (oldest_id,))

        # Insert new search history entry into the database
        cursor.execute("INSERT INTO user_images (user_id, image_path, features) VALUES (%s, %s, %s)", (user_id, uploaded_file.name, pickle.dumps(features)))
        conn.commit()

        conn.close()

        return 1
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return 0

# Feature extraction function
def feature_extraction(img_path, model):
    # Preprocess the image and extract features using the model
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommender function
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Fashion Recommender System function
# Fashion Recommender System function
def fashion_recommender(show_history=False):
    st.title('Fashion Recommender System')

    # Load precomputed features and filenames
    feature_list = np.array(pickle.load(open('feature_embeding.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

    # Load the pre-trained ResNet50 model + higher level layers
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file, st.session_state.user_id, model):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            # Extract features and get recommendations
            features = feature_extraction(uploaded_file, model)

            # Get recommendations based on the latest search
            indices_latest = recommend(features, feature_list)

            # Get recommendations based on the second latest search, if available
            if show_history:
                user_id = st.session_state.user_id
                conn = create_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT features, image_path FROM user_images WHERE user_id = %s ORDER BY id DESC LIMIT 2", (user_id,))
                user_data = cursor.fetchall()
                cursor.close()
                conn.close()

                if len(user_data) == 2:
                    user_features_latest = pickle.loads(user_data[0][0])
                    user_features_second_latest = pickle.loads(user_data[1][0])

                    # Get recommendations for the latest search
                    indices_second_latest = recommend(user_features_latest, feature_list)

                    # Display recommendations from both searches
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            if i < 3:
                                recommended_image_path = filenames[indices_latest[0][i]]
                            else:
                                recommended_image_path = filenames[indices_second_latest[0][i - 3]]

                            # Use the provided GitHub URL
                            github_url = "https://raw.githubusercontent.com/Amit-1233/Project/main/images/"
                            recommended_image_url = github_url + recommended_image_path
                            
                            st.image(recommended_image_url, use_column_width=True, caption=f"Recommendation {i+1}")
                else:
                    # Only one search history available, show 5 recommendations
                    st.info("Insufficient search history to provide recommendations from both searches.")
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            recommended_image_path = filenames[indices_latest[0][i]]

                            # Use the provided GitHub URL
                            github_url = "https://raw.githubusercontent.com/Amit-1233/Project/main/images/"
                            recommended_image_url = github_url + recommended_image_path

                            st.image(recommended_image_url, use_column_width=True, caption=f"Recommendation {i+1}")
            else:
                # Display recommendations only from the latest search
                st.subheader("Recommended Products:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(indices_latest[0]):
                            recommended_image_path = filenames[indices_latest[0][i]]

                            # Use the provided GitHub URL
                            github_url = "https://raw.githubusercontent.com/Amit-1233/Project/main/images/"
                            recommended_image_url = github_url + recommended_image_path

                            st.image(recommended_image_url, use_column_width=True, caption=f"Recommendation {i+1}")
        else:
            st.header("Some error occurred in file upload")
