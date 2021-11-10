import cv2
import  numpy as np

height=4000
fx=3793
fy=3795
cx=2073
cy=1427
focal = fx
pp = (cx,cy)
R_total = np.zeros((3, 3))
t_total = np.empty(shape=(3, 1))
cur_ROT = np.eye(3)
cur_TRS = np.zeros((3,1))
kMinNumFeature= 1500


image_id = 0
frams_arr=[]
framrs= 3

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
traj = np.zeros((600,600,3), dtype=np.uint8)

cap = cv2.VideoCapture('IMG_0189.MOV')
ret,frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ref_frm = img
P0 = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)

while(1):
    ret, new =cap.read()
    img2= cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    cur_frm = img2
    P1, st, err = cv2.calcOpticalFlowPyrLK(ref_frm, cur_frm,P0, None, **lk_params)
    E, mask = cv2.findEssentialMat(P1, P0, focal=focal, pp=pp, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, P1, P0, focal=focal, pp=pp)



    cur_TRS = cur_TRS + cur_ROT.dot(t)
    cur_ROT = R.dot(cur_ROT)
    if (P1.shape[0] < kMinNumFeature):
        P1 = cv2.goodFeaturesToTrack(cur_frm, mask = None, **feature_params)
    ref_frm =cur_frm
    P0 = P1.copy()


    cv2.imshow('img', img2)
    k = cv2.waitKey(30) & 0xff

cv2.destroyAllWindow
cap.release()