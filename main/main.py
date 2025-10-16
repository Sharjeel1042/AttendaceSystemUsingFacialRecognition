# ===============================
# FACE RECOGNITION ATTENDANCE SYSTEM
# Detection: OpenCV (Caffe SSD)
# Recognition: FaceNet (TensorFlow)
# Attendance: SQLite
# ===============================
import cv2
import os
from keras_facenet import FaceNet
from database_utils import init_database
from registration import load_dataset, register_new_student
from real_time import run_realtime_attendance


# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 'main' folder
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # go up to project root
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")




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


    while True:
        print("\n=== FACE RECOGNITION ATTENDANCE SYSTEM ===")
        print("1. Register New Student")
        print("2. Start Attendance System")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":

            register_new_student(DATASET_PATH, net, embedder, conn, database)
        elif choice == "2":
            run_realtime_attendance(net, embedder, database, conn)
        elif choice == "3":
            break
        else:
            print("[ERROR] Invalid choice. Please try again.")

        # 4. Start the real-time attendance system
        run_realtime_attendance(net, embedder, database, conn)
    # 5. Close database connection
    conn.close()
    print("[INFO] Application finished.")


if __name__ == "__main__":
    main()
