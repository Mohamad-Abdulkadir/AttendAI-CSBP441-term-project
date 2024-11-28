import os
import json
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from deepface import DeepFace
import base64
import gdown

# YOLO model
model = YOLO('models/yolov8m-face.pt')

class FaceComparisonWithDeepFace:
    def __init__(self, model_name="Facenet512"):
        self.model_name = model_name

    def get_embedding(self, image_path):
        try:
            embedding = DeepFace.represent(
                img_path=image_path, 
                model_name=self.model_name, 
                enforce_detection=False
            )
            return embedding[0]['embedding']
        except Exception as e:
            st.error(f"Error processing {image_path}: {str(e)}")
            return None

    def compare_embeddings(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

face_comparator = FaceComparisonWithDeepFace(model_name="Facenet512")

# Download images from a shared Google Drive folder
def download_images_from_drive_link(link, output_folder):
    """
    Download images from a shared Google Drive link using gdown.
    Args:
        link (str): Shared Google Drive folder link.
        output_folder (str): Local folder to save the images.
    """
    os.makedirs(output_folder, exist_ok=True)
    st.info("Downloading images from Google Drive link...")
    gdown.download_folder(link, quiet=False, output=output_folder)
    st.success(f"Images downloaded to {output_folder}")

# Preprocessing function
def preprocess_face(face_img, target_size=(224, 224)):
    resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb

# Detect faces and save them
def get_faces(folder_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    seen_faces = set()
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"Failed to read image: {img_path}")
            continue
        results = model(img)
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                face = img[y1:y2, x1:x2]
                preprocessed_face = preprocess_face(face)
                face_hash = hash(preprocessed_face.tobytes())
                if face_hash in seen_faces:
                    continue
                seen_faces.add(face_hash)
                preprocessed_filename = f"{file.split('.')[0]}_{j}.jpg"
                preprocessed_path = os.path.join(target_path, preprocessed_filename)
                cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed_face, cv2.COLOR_RGB2BGR))
    st.success("Face detection and preprocessing completed.")

# Generate the database from a Drive folder link
def generate_database(drive_link, dataset_name):
    temp_folder = f"{dataset_name}_images"
    temp_faces_folder = f"{dataset_name}_faces"
    database_path = f"{dataset_name}.json"

    # Download images from Google Drive
    download_images_from_drive_link(drive_link, temp_folder)

    # Preprocess faces and detect
    get_faces(temp_folder, temp_faces_folder)

    # Generate embeddings and save the database
    database = {}
    for file in os.listdir(temp_faces_folder):
        if file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(temp_faces_folder, file)
            student_name = os.path.splitext(file)[0]
            embedding = face_comparator.get_embedding(image_path)
            if embedding is not None:
                database[student_name] = embedding
            else:
                st.warning(f"Skipping {file} due to error in embedding generation.")
    with open(database_path, 'w') as f:
        json.dump(database, f, indent=4)
    st.success(f"Database for {dataset_name} saved to {database_path}")

def check_attendance(uploaded_files, dataset_name, threshold=0.65):
    """
    Check attendance by comparing detected faces from uploaded images with a pre-generated database.

    Args:
        uploaded_files (list): List of uploaded image files.
        dataset_name (str): Name of the dataset (class) to use for comparison.
        threshold (float): Similarity threshold for matching.

    """
    temp_folder = "temp_class_faces"
    database_path = f"{dataset_name}.json"

    if not os.path.exists(database_path):
        st.error(f"Database for {dataset_name} not found. Please generate it first.")
        return

    # Ensure temp folder exists
    os.makedirs(temp_folder, exist_ok=True)

    # Save uploaded images temporarily
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(temp_folder, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    # Load the database
    with open(database_path, 'r') as f:
        database = json.load(f)

    # Initialize attendance dictionary
    attendance = {user: {"present": False, "best_match": None} for user in database.keys()}

    # Process uploaded images: Detect, crop faces, and compare embeddings
    for file in os.listdir(temp_folder):
        if file.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(temp_folder, file)
            img = cv2.imread(image_path)

            if img is None:
                st.warning(f"Unable to read {file}. Skipping...")
                continue

            # Perform face detection on the uploaded image
            results = model(img)
            for i, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    face = img[y1:y2, x1:x2]
                    preprocessed_face = preprocess_face(face)

                    # Save cropped face temporarily for embedding generation
                    face_path = os.path.join(temp_folder, f"{file.split('.')[0]}_face_{j}.jpg")
                    cv2.imwrite(face_path, cv2.cvtColor(preprocessed_face, cv2.COLOR_RGB2BGR))

                    # Generate embedding for the cropped face
                    embedding = face_comparator.get_embedding(face_path)
                    if embedding is None:
                        st.warning(f"Unable to generate embedding for {file}. Skipping...")
                        continue

                    # Compare the embedding with all records in the database
                    best_match = None
                    best_similarity = 0
                    for user, user_embedding in database.items():
                        similarity = face_comparator.compare_embeddings(
                            embedding, np.array(user_embedding)
                        )
                        if similarity > threshold and similarity > best_similarity:
                            best_match = user
                            best_similarity = similarity

                    # Mark the best match as present
                    if best_match and not attendance[best_match]["present"]:
                        attendance[best_match] = {"present": True, "best_match": f"{file} (Face {j})"}

    # Prepare attendance results for CSV export
    attendance_list = [
        {"Name": user.split('_')[0], "Attendance": "Present" if info["present"] else "Absent"}
        for user, info in attendance.items()
    ]
    attendance_list = sorted(attendance_list, key=lambda x: x["Name"])

    # Convert to DataFrame
    df = pd.DataFrame(attendance_list)

    # Display results in Streamlit
    st.write("Attendance Results:")
    st.dataframe(df)

    # Add a download button for the CSV
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="attendance_results_{dataset_name}.csv">Download Attendance Results</a>'
    st.markdown(href, unsafe_allow_html=True)


# Streamlit UI
st.title("Face Recognition Attendance System")

options = ["Generate Database", "Check Attendance"]
choice = st.sidebar.selectbox("Select an Option", options)

if choice == "Generate Database":
    st.header("Generate Face Embeddings Database")
    drive_link = st.text_input("Google Drive Folder Link")
    dataset_name = st.text_input("Dataset Name")
    if st.button("Generate Database") and drive_link and dataset_name:
        generate_database(drive_link, dataset_name)

elif choice == "Check Attendance":
    st.header("Check Attendance")
    dataset_name = st.selectbox("Select Dataset", [f.split(".json")[0] for f in os.listdir() if f.endswith(".json")])
    uploaded_files = st.file_uploader(
        "Upload Images for Attendance Checking", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    )
    threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.65)
    if st.button("Check Attendance") and uploaded_files and dataset_name:
        check_attendance(uploaded_files, dataset_name, threshold)