import cv2
import numpy as np

def get_polygon(img):
    # get polygone from random mask shape
    img = img.astype(np.uint8)
    cnts, hiers = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
    poly = np.zeros((img.shape[0],img.shape[1]), dtype = np.uint8)
    for cnt in cnts:       
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        mask_approx = np.zeros((img.shape[0],img.shape[1]), dtype = np.uint8)
        cv2.fillPoly(mask_approx, pts =[approx], color=(255))
        poly += mask_approx
    return poly