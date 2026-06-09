import cv2
import numpy as np
import mediapipe as mp

from utils.face_detector import detect_face
from utils.blink_detector import eye_aspect_ratio
from utils.heart_rate import calculate_bpm
from utils.heart_rate import get_signal
from utils.eye_tracker import draw_eye_tracker
from utils.graph_plotter import update_graph

# -------------------------
# MediaPipe Face Mesh
# -------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------
# Webcam
# -------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

blink_count = 0
blink_state = False

LEFT_EYE = [33,160,158,133,153,144]

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh_detector.process(
        rgb_frame
    )

    bpm = 0

    # -------------------------
    # Face Detection
    # -------------------------

    faces = detect_face(frame)

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        roi = frame[
            y:y+h,
            x:x+w
        ]

        if roi.size > 0:

            green_mean = np.mean(
                roi[:, :, 1]
            )

            bpm = calculate_bpm(
                green_mean
            )

    # -------------------------
    # Face Mesh
    # -------------------------

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            draw_eye_tracker(
                frame,
                face_landmarks
            )

            h, w, _ = frame.shape

            eye_points = []

            for idx in LEFT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                eye_points.append(
                    np.array([x, y])
                )

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255,0,0),
                    -1
                )

            ear = eye_aspect_ratio(
                np.array(eye_points)
            )

            if ear < 0.20:

                if not blink_state:

                    blink_count += 1
                    blink_state = True

            else:

                blink_state = False

    # -------------------------
    # Graph Update
    # -------------------------

    signal = get_signal()

    update_graph(signal)

    # -------------------------
    # Display
    # -------------------------

    cv2.putText(
        frame,
        f"Heart Rate: {bpm} BPM",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        f"Blinks: {blink_count}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.imshow(
        "Heart Rate & Eye Tracking",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()