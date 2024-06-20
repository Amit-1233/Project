import streamlit as st
import mysql.connector
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os

# Connect to MySQL database
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
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Extract features
        features = feature_extraction(file_path, model)

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

        cursor.close()
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
def fashion_recommender(show_history=False):
    st.title('Fashion Recommender System')

    # Load precomputed features and GitHub URLs
    with open('filenames.pkl', 'rb') as f:
        github_urls = pickle.load(f)
    with open('feature_embedding.pkl', 'rb') as f:
        feature_list = np.array(pickle.load(f))

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
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

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
                                recommended_image_path = github_urls[indices_latest[0][i]]
                            else:
                                recommended_image_path = github_urls[indices_second_latest[0][i - 3]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

                else:
                    # Only one search history available, show 5 recommendations
                    st.info("Insufficient search history to provide recommendations from both searches.")
                    st.subheader("Recommended Products:")
                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        with col:
                            recommended_image_path = github_urls[indices_latest[0][i]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

            else:
                # Display recommendations only from the latest search
                st.subheader("Recommended Products:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        if i < len(indices_latest[0]):
                            recommended_image_path = github_urls[indices_latest[0][i]]
                            st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

        else:
            st.header("Some error occurred in file upload")

    elif show_history:
        # Show recommendations from search history if specified, without uploading a new image
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
            indices_latest = recommend(user_features_latest, feature_list)

            # Get recommendations for the second latest search
            indices_second_latest = recommend(user_features_second_latest, feature_list)

            # Display recommendations from both searches
            st.subheader("Based on your recent activity:")
            cols = st.columns(6)
            for i, col in enumerate(cols):
                with col:
                    if i < 3:
                        recommended_image_path = github_urls[indices_latest[0][i]]
                    else:
                        recommended_image_path = github_urls[indices_second_latest[0][i - 3]]
                    st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

        elif len(user_data) == 1:
            # Only one search history available, show 5 recommendations
            user_features_latest = pickle.loads(user_data[0][0])
            indices_latest = recommend(user_features_latest, feature_list)
            st.subheader("Based on your recent activity:")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    recommended_image_path = github_urls[indices_latest[0][i]]
                    st.image(recommended_image_path, use_column_width=True, caption=f"Recommendation {i+1}")

# Registration and authentication functions remain the same
def create_user(username, email, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO user_info (name, email, password) VALUES (%s, %s, %s)", (username, email, password))
        conn.commit()
        st.success("User created successfully!")
    except mysql.connector.Error as err:
        st.error(f"Error creating user: {err}")
    finally:
        cursor.close()
        conn.close()

def authenticate_user(email, username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_info WHERE email = %s AND name = %s AND password = %s", (email, username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def Registration():
    st.title("Registration and Login")
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

    if menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            user = authenticate_user(email, username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user[0]  # Save user ID in session state
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
    
    elif menu == "Register":
        st.subheader("Register")
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        if st.button("Register"):
            if new_password == confirm_password:
                create_user(new_username, new_email, new_password)
            else:
                st.error("Passwords do not match.")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        Registration()
    else:
        if "first_login" not in st.session_state:
            st.session_state.first_login = True

        if st.session_state.first_login:
            fashion_recommender(show_history=True)
            st.session_state.first_login = False
        else:
            fashion_recommender(show_history=False)

if __name__ == "__main__":
    main()
