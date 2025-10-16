from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import sys
from datetime import datetime

# Import your existing modules
from faceDetection import detect_faces, extract_face
from faceEmbedding import get_embedding
from database_utils import init_database, mark_attendance, add_student
from registration import load_dataset

app = Flask(__name__, template_folder='../client/templates', static_folder='../client/static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10**8)

# Load models once at startup
print("[INFO] Loading models...")
# Load face detection model
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
prototxt_path = os.path.join(MODELS_PATH, "deploy.prototxt")
model_path = os.path.join(MODELS_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load FaceNet embedder
from keras_facenet import FaceNet
embedder = FaceNet()

# Initialize database
conn = init_database()

# Load dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
database = load_dataset(DATASET_PATH, net, embedder, conn)
print(f"[INFO] Models loaded successfully! Database has {len(database)} registered faces.")


@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"[INFO] Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"[INFO] Client disconnected: {request.sid}")


@socketio.on('recognize_face')
def handle_recognize(data):
    """
    Handle face recognition request
    Expected data: {'image': 'base64_encoded_image'}
    """
    try:
        img_b64 = data.get('image')

        if not img_b64:
            emit('recognition_error', {'error': 'No image provided'})
            return

        # Remove data URL prefix if present
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        # Decode image
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('recognition_error', {'error': 'Failed to decode image'})
            return

        # Detect faces
        boxes = detect_faces(frame, net)

        recognized_students = []

        for box in boxes:
            # Extract face
            face = extract_face(frame, box)

            if face is None or face.size == 0:
                continue

            # Get embedding
            embedding = get_embedding(face, embedder)

            # Compare with database
            min_distance = float('inf')
            best_match = None

            for reg_no, stored_embedding in database.items():
                distance = np.linalg.norm(embedding - stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = reg_no

            # Threshold for recognition (0.6 works well for FaceNet)
            if min_distance < 0.6:
                # Get student info from database
                cursor = conn.cursor()
                cursor.execute("SELECT reg_no, name, semester, phone FROM students WHERE reg_no = ?", (best_match,))
                student_info = cursor.fetchone()

                if student_info:
                    # Mark attendance
                    mark_attendance(conn, best_match)

                    recognized_students.append({
                        'reg_no': student_info[0],
                        'name': student_info[1],
                        'semester': student_info[2],
                        'phone': student_info[3],
                        'confidence': float(1 - min_distance),
                        'distance': float(min_distance)
                    })

        if recognized_students:
            emit('recognition_success', {
                'students': recognized_students,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            emit('recognition_result', {
                'students': [],
                'message': 'No faces recognized'
            })

    except Exception as e:
        print(f"[ERROR] Error in recognition: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('recognition_error', {'error': str(e)})


@socketio.on('register_student')
def handle_register(data):
    """
    Handle student registration
    Expected data: {
        'reg_no': '...',
        'name': '...',
        'semester': '...',
        'phone': '...',
        'images': ['base64_image1', 'base64_image2', ...]
    }
    """
    try:
        reg_no = data.get('reg_no')
        name = data.get('name')
        semester = data.get('semester')
        phone = data.get('phone')
        images_b64 = data.get('images', [])

        if not all([reg_no, name, semester, phone, images_b64]):
            emit('registration_error', {'error': 'Missing required fields'})
            return

        # Check if student already exists
        if reg_no in database:
            emit('registration_error', {'error': f'Student with Reg No {reg_no} already exists!'})
            return

        # Decode images and extract embeddings
        embeddings = []
        for img_b64 in images_b64:
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Detect face
            boxes = detect_faces(frame, net)

            if not boxes:
                print(f"[WARN] No face detected in image, skipping.")
                continue

            # Use the largest face
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            main_box = boxes[np.argmax(areas)]

            # Extract face
            face = extract_face(frame, main_box)

            if face is not None and face.size > 0:
                # Get embedding
                embedding = get_embedding(face, embedder)
                embeddings.append(embedding)

        if len(embeddings) < 3:
            emit('registration_error', {'error': f'Could not extract enough face embeddings. Got {len(embeddings)}, need at least 3.'})
            return

        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Save to database
        add_student(conn, reg_no, name, semester, phone)

        # Update in-memory database
        database[reg_no] = avg_embedding

        # Create directory and save images (optional, for backup)
        person_dir = f"{reg_no}_{name}_{semester}_{phone}"
        person_path = os.path.join(DATASET_PATH, person_dir)

        if not os.path.exists(person_path):
            os.makedirs(person_path)

        # Save first 5 images
        for idx, img_b64 in enumerate(images_b64[:5]):
            if ',' in img_b64:
                img_b64 = img_b64.split(',')[1]
            img_bytes = base64.b64decode(img_b64)
            img_path = os.path.join(person_path, f"{idx + 1}.jpg")
            with open(img_path, 'wb') as f:
                f.write(img_bytes)

        emit('registration_success', {
            'message': f'Student {name} registered successfully with {len(embeddings)} face samples!',
            'reg_no': reg_no
        })

    except Exception as e:
        print(f"[ERROR] Error in registration: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('registration_error', {'error': str(e)})


@socketio.on('get_attendance')
def handle_get_attendance(data):
    """Get attendance records"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.reg_no, s.name, a.timestamp 
            FROM attendance a 
            JOIN students s ON a.reg_no = s.reg_no 
            ORDER BY a.timestamp DESC 
            LIMIT 100
        """)
        records = cursor.fetchall()

        attendance_list = [
            {
                'reg_no': r[0],
                'name': r[1],
                'timestamp': r[2]
            }
            for r in records
        ]

        emit('attendance_data', {'records': attendance_list})

    except Exception as e:
        print(f"[ERROR] Error fetching attendance: {str(e)}")
        emit('error', {'error': str(e)})


if __name__ == '__main__':
    print("[INFO] Starting server on http://0.0.0.0:5000")
    print("[INFO] Access from your phone using: http://YOUR_PC_IP:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
