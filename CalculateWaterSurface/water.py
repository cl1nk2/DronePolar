# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np

img_name_int = sys.argv[1]
img_name_mask = sys.argv[2]
img_name_water = sys.argv[3]

img_int = cv2.imread(img_name_int)
img_mask = cv2.imread(img_name_mask)
img_water = cv2.imread(img_name_water)

height, width, _ = img_int.shape

img_water = cv2.resize(img_water, (width, height))
img_mask = cv2.resize(img_mask, (width, height))

img_int_grey = cv2.cvtColor(img_int, cv2.COLOR_BGR2GRAY)

overexposed_threshold = 230.0
img_overexposed_mask = (img_int_grey >= overexposed_threshold).astype(np.uint8)*255

underexposed_threshold = 5.0
img_underexposed_mask = (img_int_grey <= underexposed_threshold).astype(np.uint8)*255

img_overunderexposed_mask = cv2.bitwise_or(img_overexposed_mask, img_underexposed_mask)

#cv2.imwrite('img_overexposed_mask.png', img_overexposed_mask)
#cv2.imwrite('img_underexposed_mask.png', img_underexposed_mask)
#cv2.imwrite('img_overunderexposed_mask.png', img_overunderexposed_mask)

int_mask = cv2.inRange(img_mask, np.array([0, 0, 255]), np.array([0, 0, 255]))
water_mask = cv2.inRange(img_water, np.array([255, 0, 0]), np.array([255, 0, 0]))

#cv2.imwrite('int_mask.png', int_mask)
#cv2.imwrite('water_mask.png', water_mask)

int_mask = cv2.bitwise_and(int_mask, cv2.bitwise_not(img_overunderexposed_mask))

#cv2.imwrite('int_mask_final.png', int_mask)

output = cv2.bitwise_and(int_mask, water_mask)

int_pixel_count = np.sum(int_mask==255)

if int_pixel_count > 0:
    result = np.sum(output==255)/int_pixel_count
else:
    result = 0

print(result)