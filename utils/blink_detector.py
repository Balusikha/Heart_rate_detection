import numpy as np

def eye_aspect_ratio(points):

    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])

    ear = (A + B) / (2.0 * C)

    return ear