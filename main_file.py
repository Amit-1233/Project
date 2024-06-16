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

# Function to create a connection to MySQL database
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
def save_uploaded_file(uploaded_file, user_id, model, github_urls, feature_list):
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

# Recommender function using GitHub URLs
def recommend_from_github(github_urls, features, feature_list):
    # Compute similarities with GitHub URLs and features
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    recommended_urls = []
    for idx in indices[0]:
        recommended_urls.append(github_urls[idx])

    return recommended_urls

# Streamlit app code
def main():
    st.title('Fashion Recommender System')

    # Load GitHub URLs and feature embeddings
    github_urls = pickle.load(open('filenames.pkl', 'rb'))  # Assuming 'filenames.pkl' contains GitHub URLs
    feature_list = np.array(pickle.load(open('feature_embedding.pkl', 'rb')))

    # Load the pre-trained ResNet50 model + higher level layers
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file, st.session_state.user_id, model, github_urls, feature_list):
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            # Extract features and get recommendations
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

            # Get recommendations based on GitHub URLs
            recommended_urls = recommend_from_github(github_urls, features, feature_list)

            if recommended_urls:
                st.subheader("Recommended Products:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(recommended_urls):
                            st.image(recommended_urls[i], use_column_width=True, caption=f"Recommendation {i+1}")
                        else:
                            st.warning(f"No recommendation found for index {i}.")
            else:
                st.warning("No recommendations found.")
        else:
            st.header("Some error occurred in file upload")

    st.info("To see recommendations from your recent searches, check the history option in the sidebar.")

    if st.sidebar.checkbox("Show History"):
        # Show recommendations from search history
        user_id = st.session_state.user_id
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT features, image_path FROM user_images WHERE user_id = %s ORDER BY id DESC LIMIT 2", (user_id,))
        user_data = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(user_data) >= 1:
            st.subheader("Based on your recent activity:")
            cols = st.columns(6)
            for i, col in enumerate(cols):
                with col:
                    if i < len(user_data):
                        user_features = pickle.loads(user_data[i][0])
                        indices = recommend_from_github(github_urls, user_features, feature_list)
                        recommended_image_path = github_urls[indices[0][0]]
                        if os.path.exists(recommended_image_path):
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")
                        else:
                            st.warning(f"Recommended image {i+1} not found.")
                    else:
                        st.warning(f"No recommendation found for index {i}.")
        else:
            st.info("No search history available.")

if __name__ == '__main__':
    main()
