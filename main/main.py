# ===============================
# FACE RECOGNITION ATTENDANCE SYSTEM
# Detection: OpenCV (Caffe SSD)
# Recognition: FaceNet (TensorFlow)
# Attendance: SQLite
# ===============================
import cv2
import os
from keras_facenet import FaceNet
from main.database_utils import init_database
from main.register import load_dataset
from main.realtime import run_realtime_attendance

# --- Path Configuration ---
MODELS_PATH = "models/"
DATASET_PATH = "../dataset/"


def load_models():
    """Loads the face detection and recognition models."""
    proto_path = os.path.join(MODELS_PATH, "deploy.prototxt")
    model_path = os.path.join(MODELS_PATH, "res10_300x300_ssd_iter_140000.caffemodel")

    if not (os.path.exists(proto_path) and os.path.exists(model_path)):
        print("[ERROR] Detection model files not found. Please place them in the 'models' directory.")
        exit()

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    embedder = FaceNet()
    print("[INFO] Models loaded successfully.")
    return net, embedder


def main():
    """Main function to run the attendance system."""
    # 1. Load models
    net, embedder = load_models()

    # 2. Initialize database
    conn = init_database()

    # 3. Load dataset and register known faces
    # The key is the registration number, and the value is the embedding
    database = load_dataset(DATASET_PATH, net, embedder, conn)

    if not database:
        print("[WARN] Database is empty. No known faces to recognize. Please populate the 'dataset' folder.")
        print("[INFO] Format: dataset/RegNo_Name_Semester_Phone/image.jpg")
        # You can still run the webcam to see unknown faces

    # 4. Start the real-time attendance system
    run_realtime_attendance(net, embedder, database, conn)

    # 5. Close database connection
    conn.close()
    print("[INFO] Application finished.")


if __name__ == "__main__":
    # Change directory to the 'main' folder to handle relative paths correctly
    if os.path.basename(os.getcwd()) != "main":
        os.chdir('main')
    main()
