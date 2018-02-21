from scipy.spatial import distance as dist
import dlib
import numpy as np
import cv2


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

def eye_aspect_ratio(eye):
    # comput the euclidean distances between the two sets of
    # vartical eye landmarks( x, y)- coordinates

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the suclidean distance between the horzontal
    # eye landmark (x, y)- coordinates
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)

    return ear


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

video = cv2.VideoCapture(0)

while(video.isOpened()):
    _, frame = video.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_frame, 0)

    for rect in rects:

        shape = predictor(gray_frame, rect)
        shape = shape_to_np(shape)
        left_eye = shape[42: 48]
        right_eye = shape[36: 42]

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)

        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        avgEAR = (leftEAR + rightEAR) / 2.0

        if avgEAR < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

    cv2.imshow('cam', frame)

    k = cv2.waitKey(20)

    if k == ord('q') or k == 27:
        break
    elif k == ord('p'):
        cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()