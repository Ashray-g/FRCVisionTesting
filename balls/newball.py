import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('ball.mp4')

# loop through video frames
while True:
    ret, img = cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (3, 3))

    # Define the yellow range
    lower_yellow = np.array([27, 150, 60])
    upper_yellow = np.array([33, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop through countours and draw rectangles
    for cnt in contours:

        # filter out small contours
        if cv2.contourArea(cnt) > 100:

            # get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            if float(w) / h < 1.4 and float(w) / h > 0.6:
                # draw rectangle
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
