import numpy as np
import cv2
from matplotlib import pyplot as plt
from dataset import H_dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import random

def SIFT(img1, img2 ,show = False):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    if(show):
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
        img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) 
        hmerge = np.hstack((img3, img4))
        # cv2.imwrite("figs/point.jpg", hmerge)

    return [kp1, kp2, des1,des2]

def match(imgs,sift_out,show = False):
    """
    imgs: [img1, img2]
    sift_out:[kp1,kp2,des1,des2]
    show: whether to show the matching result
    """
    img1, img2 = imgs
    kp1,kp2,des1,des2 = sift_out
    #kp: keypoints, des: descriptor
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    # filter the matches
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m]) 
    if(show):
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        # cv2.imwrite("figs/BFmatch_good.jpg", img5)
    return good

def RANSAC(matched_pairs, sift_out, k =10):
    """
    select the best 4 points
    mathed_pairs: the matched points using knn
    sift_out: the output of the sift algorithm
    k: the total iterations
    """
    """
    total_pairs=len(matched_pairs) 
    epoch=0
    best_model=None   
    besterr=np.inf
    kp1 = sift_out[0]
    kp2 = sift_out[1]
    while epoch<k:
        pairs = random.shuffle(matched_pairs)[:4]
        H_kp1 = []
        H_kp2 = []
        for i in range(4):
            trainIdx = pairs[i][0].trainIdx
            queryIdx = pairs[i][0].queryIdx
            H_kp1.append(kp1[queryIdx].pt)
            H_kp2.append(kp2[trainIdx].pt)
        
        H_kp1 = np.array(H_kp1)
        H_kp2 = np.array(H_kp2)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for i in range(4):
            img1 = cv2.circle(img1,(int(H_kp1[i][0]),int(H_kp1[i][1])),color=(255,0,255),radius=1)
            img2 = cv2.circle(img2,(int(H_kp2[i][0]),int(H_kp2[i][1])),color=(255,0,255),radius=1) 
        hmerge = np.hstack((img1, img2))
        cv2.imwrite("figs/mached_4point.jpg", hmerge)    
        epoch+=1
    if best_model is None:
        raise ValueError("无法拟合出model")
    else:
    	return best_model,besterr
    """
    """temperal method: use the interface provided by opencv to do RANSAC"""
    kp1 = sift_out[0]
    kp2 = sift_out[1]
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in matched_pairs ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in matched_pairs ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    return H,matchesMask


def get_error(model,test,y_true):
    y_predict=model.predict(test)
    error=np.sqrt((y_predict-y_true)**2)
    mean_error=np.mean(error)
    return error,mean_error

def compute_H(points):
    """
    points:
    [
        [(x0,y0),(x1,y1),(x2,y2),(x3,y3)],
        [(x0',y0'),(x1',y1'),(x2',y2'),(x3',y3')]
    ]
    return: H 
    """
    H = cv2.getPerspectiveTransform(np.float32(points[0]), \
    np.float32(points[1]))
    return H  

def warp(img, H):
    warp_img = cv2.warpPerspective(img, H, (img.shape[1] , img.shape[0]))
    return warp_img

def sift_ransac(img_pair):
    img1 = img_pair[0]
    img2 = img_pair[1]
    sift_out = SIFT(img1,img2,True)
    matches = match([img1,img2],sift_out,True)
    H,_ = RANSAC(matches,sift_out)

    sample = np.zeros((128,128*3),dtype=np.uint8)
    sample[:,:128] = img1
    sample[:,128:256] = img2
    warped = warp(img1,H)
    sample[:,256:128*3] = warped
    cv2.imwrite("sample.jpg",sample)

    return H

    
if __name__ == "__main__":
    """ for quick validate"""
    val_dataset =  H_dataset("./datasets",train=False,norm=False)
    image_pairs,label = val_dataset[0]
    img1 = image_pairs[0]
    img2 = image_pairs[1]
    sift_out = SIFT(img1,img2,True)
    matches = match([img1,img2],sift_out,True)
    H,_ = RANSAC(matches,sift_out)
    warped = warp(img1,H)

    sample = np.zeros((128,128*3),dtype=np.uint8)
    sample[:,:128] = img1
    sample[:,128:256] = img2
    sample[:,256:128*3] = warped
    cv2.imwrite("sample.jpg",sample)

