import cv2, numpy as np
from pathlib import Path

def verify(imgA_path, imgB_path, min_inliers=15):
    img1 = cv2.imread(str(imgA_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(imgB_path), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None: return False, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < min_inliers: return False, len(good)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers >= min_inliers, inliers
