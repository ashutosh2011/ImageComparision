# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 23:42:30 2018

@author: Ashutosh
"""

# 2017.12.22 15:48:03 CST
# 2017.12.22 16:00:14 CST
import cv2
import numpy as np
#from PIV import Image

#im1 = Image.open("C:/Users/Gauranga/Pictures/screenshotOne.png")
#im2 = Image.open("C:/Users/Gauranga/Pictures/screenshotTwo.png")

img1 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss1.png")
img2 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss2.png")

#img1 = cv2.imread("C:/Users/Gauranga/Des/screenshotTwo.png")
#img2 = cv2.imread("C:/Users/Gauranga/Pictures/screenshotTwo.png")


diff = cv2.absdiff(img1, img2)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

th = 1
imask =  mask>th

canvas = np.zeros_like(img2, np.uint8)
canvas[imask] = img2[imask]

cv2.imshow("result.png", canvas)

#difference = cv2.subtract(img1, img2)
#
#result = not np.any(difference) #if difference is all zeros it will return False
#
#if result is True:
#    print ("The images are the same")
#else:
#    cv2.imwrite("result.jpg", difference)
#    print ("the images are different")