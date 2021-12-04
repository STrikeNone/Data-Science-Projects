import cv2
import numpy
import numpy as np

cap = cv2.VideoCapture("IR_walking.mp4")

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 128    # the graylevel of images
params.maxThreshold = 200
params.filterByColor = True
params.blobColor = 255
# Filter by Area
params.filterByArea = False
params.minArea = 1000
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = True
params.minCircularity = 0.7

detector = cv2.SimpleBlobDetector_create(params)

while True:
    _, frame = cap.read()
    blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, blur = cv2.threshold(blur, 90, 255, cv2.THRESH_TOZERO)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blur = cv2.erode(blur, element, iterations=1)
    blur = cv2.dilate(blur, element, iterations=1)
    blur = cv2.morphologyEx(blur, cv2.MORPH_ELLIPSE, np.ones((5, 5), np.uint8))
    blur = cv2.medianBlur(blur, 9)
    keypoints = detector.detect(blur)

    for i in range(1, len(keypoints)):
        x, y = np.int64(keypoints[i].pt[0]), np.int64(keypoints[i].pt[1])
        sz = np.int64(keypoints[i].size)
        if sz > 1:
            sz = np.int64(sz / 2)
        # notice there's no boundary check for pt1 and pt2, you have to do that yourself
        kp = cv2.rectangle(frame, (x - sz, y - sz), (x + sz, y + sz), color=(0, 255, 0), thickness=2)

 #   kp = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("F",frame)
    key = cv2.waitKey(10)
    if key == 5:
        break

cap.release()
cv2.destroyAllWindows()