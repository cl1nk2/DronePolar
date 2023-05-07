# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np

img_name_int = sys.argv[1]
img_name_water = sys.argv[2]

img_int = cv2.imread(img_name_int)
img_water = cv2.imread(img_name_water)

height1, width1, _ = img_int.shape

img_water = cv2.resize(img_water, (width1, height1))

int_mask = cv2.inRange(img_int, np.array([0, 0, 255]), np.array([0, 0, 255]))
water_mask = cv2.inRange(img_water, np.array([255, 0, 0]), np.array([255, 0, 0]))

output = cv2.bitwise_and(int_mask, water_mask)

int_pixel_count = np.sum(int_mask==255)

if int_pixel_count > 0:
    result = np.sum(output==255)/int_pixel_count
else:
    result = 0

print(result)