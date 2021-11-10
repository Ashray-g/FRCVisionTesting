import cv2

cap = cv2.VideoCapture("IMG_0199.MOV")

ret, old_img = cap.read()

feature_params = dict(maxCorners=400, qualityLevel=0.5, minDistance=3, blockSize=7)

lk_params = dict(
    winSize=(40, 40),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,
                0.03),
)

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

    for i in p0:
        x = int(i[0][0])
        y = int(i[0][1])
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 2)

    cv2.imshow('optical flow', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
