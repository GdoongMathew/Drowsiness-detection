from scipy.spatial import distance as dist
from threading import Thread
import matplotlib.pyplot as plt
import playsound
import dlib
import numpy as np
import cv2


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False

list_lenLimit = 150
EAR_Tol = []

'''
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = plt.plot(EAR_Tol, 'b')
'''

def play_alarm(path):
    # play the alarm sound from the given path
    print(path)
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    # comput the euclidean distances between the two sets of
    # vartical eye landmarks( x, y)- coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the suclidean distance between the horzontal
    # eye landmark (x, y)- coordinates
    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2 * c)

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
        EAR_Tol.append(avgEAR)

        #print(avgEAR)

        if avgEAR < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    t = Thread(target=play_alarm,
                               args=('Alarm01.wav',))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

    '''
        plt.title('EAR')
        plt.plot(EAR_Tol, 'b')
    plt.pause(0.0001)
    '''
    cv2.imshow('cam', frame)

    if len(EAR_Tol) >= list_lenLimit:
        EAR_Tol = EAR_Tol[-list_lenLimit:]

    k = cv2.waitKey(20)

    if k == ord('q') or k == 27:
        break
    elif k == ord('p'):
        cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()
#plt.close()
