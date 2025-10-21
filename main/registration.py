import cv2
import numpy as np
import os
from faceDetection import detect_faces, extract_face
from faceEmbedding import get_embedding
from database_utils import add_student


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


def register_new_student(dataset_path, net, embedder, conn, database):
    """
    Captures images from webcam to register a new student.
    """
    print("\n=== REGISTER NEW STUDENT ===")

    # Get student details

    reg_no = input("Enter Registration Number: ").strip()
    name = input("Enter Name: ").strip()
    semester = input("Enter Semester: ").strip()
    phone = input("Enter Phone Number: ").strip()

    if not all([reg_no, name, semester, phone]):
        print("[ERROR] All fields are required!")
        return

    # Check if student already exists
    if reg_no in database:
        print(f"[ERROR] Student with Reg No {reg_no} already exists!")
        return

    # Create directory for the student
    person_dir = f"{reg_no}/{name}/{semester}/{phone}"
    person_path = os.path.join(dataset_path, person_dir)

    if not os.path.exists(person_path):
        os.makedirs(person_path)

    # Capture images from webcam
    print("[INFO] Starting webcam... Press SPACE to capture (need 5 images), ESC to cancel.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    captured_images = []
    image_count = 0

    while image_count < 5:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Display instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captured: {image_count}/5", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Register New Student", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE key
            image_path = os.path.join(person_path, f"{image_count + 1}.jpg")
            cv2.imwrite(image_path, frame)
            captured_images.append(image_path)
            image_count += 1
            print(f"[INFO] Image {image_count}/5 captured.")

        elif key == 27:  # ESC key
            print("[INFO] Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # Process the captured images
    print("[INFO] Processing captured images...")
    avg_embedding = register_person(captured_images, net, embedder)

    if avg_embedding is not None:
        database[reg_no] = avg_embedding
        add_student(conn, reg_no, name, semester, phone)
        print(f"[SUCCESS] Student {name} registered successfully!")
    else:
        print("[ERROR] Failed to register student. No valid face embeddings found.")
        # Clean up the directory if registration failed
        import shutil
        if os.path.exists(person_path):
            shutil.rmtree(person_path)



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

