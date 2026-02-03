import cv2
import matplotlib.pyplot as plt
from processing import utilities as filters
import numpy as np


img = cv2.imread("D:/Python/wbc-counter/static/uploads/m.png")
s = filters.extract_channels(img)
m = filters.median_blur(s, kernel_size=5)
b = filters.threshold(m, 90)

count, f = filters.filter_and_count_manual(b, min_area=400)
print(count)
cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1", 1400, 800)
cv2.namedWindow("2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2", 1400, 800)
# cv2.namedWindow("3", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("3", 1400, 800)
# # cv2.namedWindow("4", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("4", 1400, 800)
cv2.imshow('1', b)
cv2.imshow('2', f)
# cv2.imshow('3', s)
# # cv2.imshow('4', img)
#
cv2.waitKey(0)
cv2.destroyAllWindows()
