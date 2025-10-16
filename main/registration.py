import cv2
import numpy as np
import os
from .face_detection import detect_faces, extract_face
from .face_embedding import get_embedding
from .database_utils import add_student


def register_person(image_paths, net, embedder):
    """
    Computes the average embedding for a person from a list of their images.
    """
    embeddings = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] Could not read image {path}, skipping.")
            continue

        boxes = detect_faces(frame, net)
        if not boxes:
            print(f"[WARN] No face detected in {path}, skipping.")
            continue

        # In case of multiple faces, use the one with the largest area
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        main_box = boxes[np.argmax(areas)]

        face = extract_face(frame, main_box)
        emb = get_embedding(face, embedder)
        embeddings.append(emb)

    if not embeddings:
        return None

    # Return the average of all embeddings for robustness
    return np.mean(embeddings, axis=0)


def load_dataset(dataset_path, net, embedder, conn):
    """
    Loads images from the dataset folder, registers each person,
    and returns a dictionary of their embeddings.
    """
    print("[INFO] Loading dataset and registering known faces...")
    database = {}
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return database

    # Loop through each sub-directory (each person) in the dataset folder
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            try:
                # The folder name is expected to be 'RegNo_Name_Semester_Phone'
                reg_no, name, semester, phone = person_dir.split('_')
            except ValueError:
                print(f"[WARN] Skipping directory with incorrect format: {person_dir}.")
                continue

            # Find all image files for the person
            image_files = [os.path.join(person_path, f) for f in os.listdir(person_path) if
                           f.endswith(('.jpg', '.png', '.jpeg'))]

            if not image_files:
                print(f"[WARN] No images found for {name}, skipping.")
                continue

            # Get the average embedding for the person
            avg_embedding = register_person(image_files, net, embedder)

            if avg_embedding is not None:
                database[reg_no] = avg_embedding
                # Add the student's details to the database
                add_student(conn, reg_no, name, semester, phone)

    print("[INFO] All known faces have been processed.")
    return database

