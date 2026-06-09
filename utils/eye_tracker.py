import cv2

LEFT_IRIS = [474, 475, 476, 477]

def draw_eye_tracker(frame, landmarks):

    h, w, _ = frame.shape

    iris_points = []

    for idx in LEFT_IRIS:

        lm = landmarks.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        iris_points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 255, 255),
            -1
        )

    if len(iris_points) > 0:

        center_x = int(
            sum([p[0] for p in iris_points])
            / len(iris_points)
        )

        center_y = int(
            sum([p[1] for p in iris_points])
            / len(iris_points)
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "Eye Tracking",
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )