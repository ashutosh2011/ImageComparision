# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:30:41 2018

@author: Ashutosh
"""

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pytesseract import image_to_string
import jellyfish
import pandas as pd
from skimage.measure import compare_ssim as ssim 
from skimage import color
import math

width = 1000
height = 1000

def resize(filename):
    img = cv2.imread(filename)
    #change this for better result
    
    return cv2.resize(img, (width, height))     

def featureDetection (filename1, filename2,a):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    
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
    score = []

    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])            
            score.append(math.exp((-a)*m.distance))
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:50:2],None,flags=2)
#    
    plt.imshow(img3),plt.show()
#    print("Percentage match = ", len(good)/min(len(des1), len(des2)) * 100 )
#    return len(good)/min(len(des1), len(des2)) * 100, sum(score)/len(score) *100
    return sum(score)/len(score) *100

def histogram(filename1, filename2):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    if(img1.size != img2.size):    
        img1 = resize(filename1)
        img2 = resize(filename2)
    hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
    
    hist = abs(hist1 - hist2)
    return sum(hist)[0]/ max(sum(hist1)[0], sum(hist2)[0]) * 100

def mse(filename1, filename2):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    if(img1.size != img2.size):    
        img1 = resize(filename1)
        img2 = resize(filename2)
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err = err ** (0.5) * 100
    err /= float(img1.shape[0] * img2.shape[1]) 
    return err

def sim(filename1, filename2):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    if(img1.size != img2.size): 
        img1 = resize(filename1)
        img2 = resize(filename2)
    img1 = color.rgb2gray(img1)
    img2 = color.rgb2gray(img2)
    return ssim(img1, img2, full=True)[0]*100
    
if __name__ == "__main__":
    
    baseloc = "C:/Users/Gauranga/Pictures/fisv/"
    #baseloc = "./fisv/"
    baseimg = baseloc + "baseimg.png"
    happy = baseloc + "happy.png"
    fail = baseloc + "deffail.png"
    partialFail = baseloc + "fail.png"
    subtleFail = baseloc + "partial fail.png"
    TextMismatch = baseloc + "textFail.png"
    iPadPro = baseloc + "iPad pro.png"
    iPad = baseloc + "iPad.png"
    IphoneX = baseloc + "iPhone X.png"
    iPhone6P = baseloc + "iPhone 6_7_8 Plus.png"
    iPhone6 = baseloc + "iPhone 6_7_8.png"
    iPhone5 = baseloc + "iPhone 5_SE.png"
    Pixel2XL = baseloc + "Pixel 2 XL.png"
    pixel2 = baseloc + "Pixel 2.png"
    galaxys5 = baseloc + "Galaxy S5.png"
    
    #adding some some sample data
    simg1 = ["macelc_chrome_38.0.png","macelc_chrome_40.0.png","macelc_chrome_41.0.png","macelc_chrome_44.0.png","macelc_chrome_45.0.png","macelc_chrome_49.0.png","macelc_chrome_50.0.png"]
    
    fdt = []
    mset = []
    simt = []
    hist = []
    namet = []
    filename = []
    imgname = [happy, fail, partialFail, subtleFail, TextMismatch]

    for qimg in imgname:     
        fd = featureDetection(baseimg, qimg,0.05)
        testname = qimg[7:-4]
        fnamei = qimg
        msei = mse(baseimg, qimg)
        histi = histogram(baseimg, qimg)
        simi = sim(baseimg, qimg)
        print("\n\n=============Test case "+ testname+"================\n")
        print("\n", qimg)
        print("\n Feature detection : ", fd)
        print("\n Square error :", msei)
        print("\n SSIM :", simi)
        print("\n Histogram :", histi)
        fdt.append(fd)
        filename.append(fnamei)
        hist.append(histi)
        simt.append(simi)
        namet.append(testname)
        mset.append(msei)
        
    for qimg in simg1: 
        testname = qimg[:-4]
        qimg = baseloc + "archive_1ea20f065389efec400591260a6ad92fd985e147/" + qimg
        baseimg = baseloc + "archive_1ea20f065389efec400591260a6ad92fd985e147/" + simg1[0]
        fd = featureDetection(baseimg, qimg,0.001)
        fnamei = qimg
        msei = mse(baseimg, qimg)
        histi = histogram(baseimg, qimg)
        simi = sim(baseimg, qimg)
        print("\n\n=============Test case "+ testname+"================\n")
        print("\n", qimg)
        print("\n Feature detection : ", fd)
        print("\n Square error :", msei)
        print("\n SSIM :", simi)
        print("\n Histogram :", histi)
        fdt.append(fd)
        filename.append(fnamei)
        hist.append(histi)
        simt.append(simi)
        namet.append(testname)
        mset.append(msei)
    width = 500
    height = 900
    print("\n\n\n=============Test case Phones================\n")
    phones = [IphoneX, iPhone6P,iPhone6,iPhone5,Pixel2XL,pixel2,galaxys5]
    for qimg in phones:
        fd = featureDetection(iPhone6, qimg,0.001)
        testname = qimg[7:-4]
        fnamei = qimg
        msei = mse(iPhone6, qimg)
        histi = histogram(iPhone6, qimg)
        simi = sim(iPhone6, qimg)
        print("\n\n=============Test case "+ testname+"================\n")
        print("\n", qimg)
        print("\n Feature detection : ", fd)
        print("\n Square error :", msei)
        print("\n SSIM :", simi)
        print("\n Histogram :", histi)
        fdt.append(fd)
        filename.append(fnamei)
        hist.append(histi)
        simt.append(simi)
        namet.append(testname)
        mset.append(msei)

    print("\n\n=============Test case iPad================\n")
    baseimg = iPad
    qimg = iPadPro
    fd = featureDetection(baseimg, qimg,0.001)
    testname = qimg[7:-4]
    fnamei = qimg
    msei = mse(baseimg, qimg)
    histi = histogram(baseimg, qimg)
    simi = sim(baseimg, qimg)
    print("\n", qimg)
    print("\n Feature detection : ", fd)
    print("\n Square error :", msei)
    print("\n SSIM :", simi)
    print("\n Histogram :", histi)
    fdt.append(fd)
    filename.append(fnamei)
    hist.append(histi)
    simt.append(simi)
    namet.append(testname)
    mset.append(msei)
    
df = pd.DataFrame({'Name': namet,
                       'File Name': filename,
                       'Feature Detection': fdt,
                       'Histogram': hist,
                       'SSIM': simt,
                       'Square Error': mset})
df.to_csv("result2.csv", sep='\t')    
