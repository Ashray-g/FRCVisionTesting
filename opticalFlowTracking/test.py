import cv2
import time
import numpy as np

cap = cv2.VideoCapture("IMG_0199.MOV")

ret, old_img = cap.read()

feature_params = dict(maxCorners=400, qualityLevel=0.3, minDistance=3, blockSize=7)

lk_params = dict(
    winSize=(40, 40),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                0.03),
)

old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)

old_p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

self_r = np.zeros(shape=(3, 3))
self_t = np.zeros(shape=(3, 3))

focal = 1.0
pp = (0, 0)

v =0

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, gray, old_p0, None, **lk_params
    )

    good_new = p1[st > 0.7]
    good_old = old_p0[st > 0.7]

    for i, (g1, g2) in enumerate(zip(good_new, good_old)):
        x1 = int(g1[0])
        y1 = int(g1[1])
        x = int(g2[0])
        y = int(g2[1])
        cv2.line(img, (x1, y1), (x, y), (0, 255, 0), 2)
        # cv2.line(img, (x1, y1), (x, y), (0, 0, 0), 2)
        cv2.rectangle(img, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), 2)
        # cv2.rectangle(img, (x1 - 1, y1 - 1), (x1 + 3, y1 + 1), (0, 255, 0), 2)

    cv2.imshow('optical flow', img)

    old_p0 = p0
    old_gray = gray

    v = v + 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
