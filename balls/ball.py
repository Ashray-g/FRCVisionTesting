import cv2
import numpy as np


img = cv2.imread("ballz1.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the yellow range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the HSV image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

# Find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#loop through countours and draw rectangles
for cnt in contours:
    #filter out small contours  
    if cv2.contourArea(cnt) > 100:
    
        #get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        if w/h < 1.1 or w/h > 0.9:
            #draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h+10), (0, 255, 0), 2)





# Find the largest contour
# largest_contour = max(contours, key=cv2.contourArea)

# Find the radius of the ball
# ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

#bounding box around largest contour
# x, y, w, h = cv2.boundingRect(largest_contour)
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("mask", img)
cv2.waitKey(1)
cv2.destroyAllWindows()
