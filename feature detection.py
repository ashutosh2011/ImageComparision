# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:24:41 2018

@author: Ashutosh
"""
'''========================1==================================='''
#import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#import numpy as np
#import cv2
#
#
#print("img read")
##
##img1 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotOne.png',0)     
##img2 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotTwo.png',0) 
#
#img1 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss1.png")
#img2 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss2.png")
#
#print("orb")
## Initiate SIFT detector
#orb = cv2.ORB_create()
#
#print("detect and compute")
## find the keypoints and descriptors with SIFT
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)
#
#print("matching")
## create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
## Match descriptors.
#matches = bf.match(des1,des2)
#
## Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
#
## Draw first 10 matches.
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
#
#plt.imshow(img3),plt.show()

'''=================================2================================'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

#img1 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotOne.png',0)          # queryImage
#img2 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotTwo.png',0) # trainImage

img1 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss1.png")
img2 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss2.png")

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()

print("Percentage match = ", len(good)/min(len(des1), len(des2)) * 100 )

'''====================================3========================================='''

#import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#
##img1 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotOne.png',0)          # queryImage
##img2 = cv2.imread('C:/Users/Gauranga/Pictures/screenshotTwo.png',0) # trainImage
#
#img1 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss1.png")
#img2 = cv2.imread("C:/Users/Gauranga/Desktop/screenshots/ss2.png")
#
## Initiate SIFT detector
#freak = cv2.xfeatures2d.FREAK_create()
#
## find the keypoints and descriptors with SIFT
#kp1, des1 = freak.detectAndCompute(img1,None)
#kp2, des2 = freak.detectAndCompute(img2,None)
#
## BFMatcher with default params
#bf = cv2.BFMatcher()
#matches = bf.knnMatch(des1,des2, k=2)
#
## Apply ratio test
#good = []
#for m,n in matches:
#    if m.distance < 0.75*n.distance:
#        good.append([m])
#
## cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#
#plt.imshow(img3),plt.show()

'''===========================misc=============================================='''
#from PIL import Image
#from PIL import ImageChops
#
#im1 = Image.open("C:/Users/Gauranga/Pictures/screenshotOne.png")
#im2 = Image.open("C:/Users/Gauranga/Pictures/screenshotTwo.png")
#
#diff = ImageChops.difference(im2, im1)

'''=========================================4==================================='''

import cv2
#import sys
#import os.path
#import numpy as np
#
#def drawMatches(img1, kp1, img2, kp2, matches):
#
#    rows1 = img1.shape[0]
#    cols1 = img1.shape[1]
#    rows2 = img2.shape[0]
#    cols2 = img2.shape[1]
#
#    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
#    out[:rows1,:cols1] = np.dstack([img1])
#    out[:rows2,cols1:] = np.dstack([img2])
#    for mat in matches:
#        img1_idx = mat.queryIdx
#        img2_idx = mat.trainIdx
#        (x1,y1) = kp1[img1_idx].pt
#        (x2,y2) = kp2[img2_idx].pt
#
#        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
#        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
#        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)
#
#    return out
#
#def compare(filename1, filename2):
#    img1 = cv2.imread(filename1)          # queryImage
#    img2 = cv2.imread(filename2)          # trainImage
#
#    # Initiate SIFT detector
#    sift = cv2.xfeatures2d.SIFT_create()
#
#    # find the keypoints and descriptors with SIFT
#    kp1, des1 = sift.detectAndCompute(img1,None)
#    kp2, des2 = sift.detectAndCompute(img2,None)
#
#    # BFMatcher with default params
#    bf = cv2.BFMatcher()
#    matches = bf.match(des1,des2)
#
#    matches = sorted(matches, key=lambda val: val.distance)
#
#    img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
#
#    # Show the image
#    cv2.imshow('Matched Features', img3)
#    cv2.waitKey(0)
#    cv2.destroyWindow('Matched Features')
##
##if len(sys.argv) != 3:
##    sys.stderr.write("usage: compare.py <queryImageFile> <sourceImageFile>\n")
##    sys.exit(-1)
#    
#compare('C:/Users/Gauranga/Desktop/screenshots/ss1.png', 'C:/Users/Gauranga/Desktop/screenshots/ss1.png')
