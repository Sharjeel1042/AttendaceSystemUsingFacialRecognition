# ===============================
# FACE RECOGNITION ATTENDANCE SYSTEM
# Detection: OpenCV (Caffe SSD)
# Recognition: FaceNet (TensorFlow)
# Attendance: SQLite
# ===============================

import cv2
import numpy as np
from keras_facenet import FaceNet
from numpy.linalg import norm
import sqlite3
from datetime import datetime


# -------------------------------
# 1. Load Models
# -------------------------------
def load_models():
    proto_path = "models/deploy.prototxt"  # Download from OpenCV repo
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"  # Download from OpenCV repo
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    embedder = FaceNet()
    print("[INFO] Models loaded successfully.")
    return net, embedder


# -------------------------------
# 2. Face Detection
# -------------------------------
def detect_faces(frame, net, confidence_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype("int"))
    return boxes


# -------------------------------
# 3. Face Extraction
# -------------------------------
def extract_face(frame, box):
    (x1, y1, x2, y2) = box
    x1, y1 = max(0, x1), max(0, y1)
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))
    return face


# -------------------------------
# 4. Face Embedding
# -------------------------------
def get_embedding(face, embedder):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face)[0]
    return embedding


# -------------------------------
# 5. Cosine Similarity
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# -------------------------------
# 6. Register New Person
# -------------------------------
def register_person(name, image_paths, net, embedder, database):
    embeddings = []
    for path in image_paths:
        frame = cv2.imread(path)
        boxes = detect_faces(frame, net)
        if len(boxes) == 0:
            print(f"[WARN] No face detected in {path}")
            continue
        face = extract_face(frame, boxes[0])
        emb = get_embedding(face, embedder)
        embeddings.append(emb)

    if len(embeddings) == 0:
        print(f"[ERROR] No valid faces found for {name}")
        return

    avg_emb = np.mean(embeddings, axis=0)
    database[name] = avg_emb
    print(f"[INFO] Registered {name} successfully.")


# -------------------------------
# 7. Recognize Person
# -------------------------------
def recognize_person(frame, net, embedder, database, threshold=0.6):
    boxes = detect_faces(frame, net)
    if len(boxes) == 0:
        return None, None, frame

    for box in boxes:
        face = extract_face(frame, box)
        emb = get_embedding(face, embedder)

        best_name, best_score = None, -1
        for name, db_emb in database.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_name, best_score = name, score

        (x1, y1, x2, y2) = box
        color = (0, 255, 0) if best_score > threshold else (0, 0, 255)
        label = best_name if best_score > threshold else "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return best_name if best_score > threshold else None, best_score, frame


# -------------------------------
# 8. Attendance Database
# -------------------------------
def init_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    print("[INFO] Database initialized.")
    return conn


def mark_attendance(name, conn):
    if not name:
        return
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, now))
    conn.commit()
    print(f"[ATTENDANCE] {name} marked at {now}")


# -------------------------------
# 9. Real-time Attendance Loop
# -------------------------------
def run_realtime_attendance(net, embedder, database, conn):
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, score, frame = recognize_person(frame, net, embedder, database)
        if name:
            mark_attendance(name, conn)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# 10. Main Program
# -------------------------------
if __name__ == "__main__":
    net, embedder = load_models()
    conn = init_database()
    database = {}

    # Register people (provide your own image paths)
    register_person("Sharjeel", ["sharjeel1.jpg", "sharjeel2.jpg"], net, embedder, database)
    register_person("Ali", ["ali1.jpg", "ali2.jpg"], net, embedder, database)

    # Start real-time recognition
    run_realtime_attendance(net, embedder, database, conn)
