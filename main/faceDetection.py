import cv2
import numpy as np

def detect_faces(frame, net, confidence_threshold=0.5):
    """
    Detects faces in an image frame using a pre-trained Caffe model.

    Args:
        frame (numpy.ndarray): The input image frame.
        net (cv2.dnn_Net): The loaded face detection model.
        confidence_threshold (float): The minimum probability to filter weak detections.

    Returns:
        list: A list of bounding boxes for detected faces.
    """
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

def extract_face(frame, box):
    """
    Extracts a face from the frame using the bounding box coordinates.

    Args:
        frame (numpy.ndarray): The input image frame.
        box (tuple): The bounding box coordinates (startX, startY, endX, endY).

    Returns:
        numpy.ndarray: The cropped and resized face image (160x160).
    """
    (x1, y1, x2, y2) = box
    x1, y1 = max(0, x1), max(0, y1)
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))
    return face

