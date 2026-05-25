import cv2
import numpy as np

img = cv2.imread("image.jpg")
Z = img.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
segmented = res.reshape(img.shape)

cv2.imshow("Segmented", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()