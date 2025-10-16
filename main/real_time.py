import cv2
from recognition import recognize_person
from database_utils import mark_attendance


def run_realtime_attendance(net, embedder, database, conn):
    """
    Starts the real-time attendance loop using the webcam.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting webcam... Press 'q' to quit.")
    cv2.namedWindow("Attendance System")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Recognize a person in the current frame
        reg_no, processed_frame = recognize_person(frame, net, embedder, database, conn)

        # If a registered person is recognized, mark their attendance
        if reg_no:
            mark_attendance(conn, reg_no)

        # Display the processed frame
        cv2.imshow("Real-Time Attendance System", processed_frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")

