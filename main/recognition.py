import cv2
import numpy as np
from numpy.linalg import norm
from .face_detection import detect_faces, extract_face
from .face_embedding import get_embedding


def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b))


def recognize_person(frame, net, embedder, database, conn, threshold=0.6):
    """
    Recognizes faces in a frame by comparing them to a database of known embeddings.

    Args:
        frame: The video frame.
        net: Face detection model.
        embedder: FaceNet model.
        database (dict): Dictionary of known embeddings {reg_no: embedding}.
        conn: SQLite database connection.
        threshold (float): Similarity threshold for recognition.

    Returns:
        tuple: (recognized registration number, annotated frame)
    """
    boxes = detect_faces(frame, net)
    recognized_reg_no = None

    for box in boxes:
        face = extract_face(frame, box)
        emb = get_embedding(face, embedder)

        best_reg_no, best_score = None, -1
        # Find the best match in the database
        for reg_no, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_reg_no, best_score = reg_no, score

        (x1, y1, x2, y2) = box

        # Check if the best match is above the confidence threshold
        if best_score > threshold:
            color = (0, 255, 0)  # Green for recognized
            recognized_reg_no = best_reg_no

            # Fetch student's name from the database for display
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM students WHERE reg_no = ?", (recognized_reg_no,))
            result = cursor.fetchone()
            name = result[0] if result else "Name N/A"
            label = f"{name} ({recognized_reg_no})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return recognized_reg_no, frame

