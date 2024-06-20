import mysql.connector
import streamlit as st
import pickle
import os

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

def save_uploaded_file(uploaded_file, user_id, model):
    try:
        from main_file import feature_extraction

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

def get_user_images(user_id, limit=2):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT features, image_path FROM user_images WHERE user_id = %s ORDER BY id DESC LIMIT %s", (user_id, limit))
    user_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return user_data
