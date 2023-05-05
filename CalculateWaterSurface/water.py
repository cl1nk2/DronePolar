# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np

img_name_int = sys.argv[1]
img_name_water = sys.argv[2]

img_int = cv2.imread(img_name_int)
img_water = cv2.imread(img_name_water)

int_mask = cv2.inRange(img_int, np.array([0, 0, 255]), np.array([0, 0, 255]))
water_mask = cv2.inRange(img_water, np.array([255, 0, 0]), np.array([255, 0, 0]))

output = cv2.bitwise_and(int_mask, water_mask)

result = np.sum(output==255)/np.sum(int_mask==255)

print(result)