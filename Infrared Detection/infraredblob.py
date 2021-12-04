import cv2
import numpy as np

img = cv2.imread("IR3.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.7, fy=0.7)
'''
#edges = cv2.Canny(img, 150, 200)
#blur = cv2.medianBlur(img, 17)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.erode(img, element, iterations=2)
mask = cv2.medianBlur(img, 5)
mask = cv2.dilate(mask, element, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
mask = cv2.medianBlur(img, 15)

_, mask = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
mask = cv2.dilate(mask, element, iterations=2)
mask = cv2.erode(mask, element, iterations=3)
mask = cv2.dilate(mask, element, iterations=1)
mask = cv2.medianBlur(mask, 15)
'''
# Using a Colour Range Filter
img = cv2.imread("IR3.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)
mask = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
white = (145, 60, 255)
gray = (0, 0, 200)
mask = cv2.inRange(mask, gray, white)
mask = cv2.bitwise_and(img, img, mask=mask)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 200  # the graylevel of images
params.maxThreshold = 255
params.filterByColor = True
params.blobColor = 255
# Filter by Area
params.filterByArea = True
params.minArea = 10
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = True
params.minCircularity = 0.1
params.maxCircularity = 0.785

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(mask)

kp = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Blurred", kp)
cv2.waitKey(0)
