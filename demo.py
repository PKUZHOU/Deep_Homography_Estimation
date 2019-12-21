import argparse
import torch
import torch.autograd.function as F
from dataset import H_dataset
import argparse
from network import get_model
from sift_ransac import sift_ransac
import numpy as np
import cv2

def SIFT_stitch(images):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        # match features between the two images
        imageA = images[0]
        imageB = images[1]
        imageA_gray = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
        imageB_gray = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
        H = sift_ransac([imageA_gray,imageB_gray])
        result = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        return result

def CNN_stitch(images,trained_model):
    imageA = images[0]
    imageB = images[1]

    model = get_model(pretrained=True)
    model.load_state_dict(torch.load(trained_model))

    min_w = min(imageB.shape[0],imageB.shape[1])
    scale = min_w/128.
    imageA = imageA[imageA.shape[0]//2-min_w//2:imageA.shape[0]//2+min_w//2,\
        imageA.shape[1]//2-min_w//2:imageA.shape[1]//2+min_w//2]
    imageB = imageB[imageB.shape[0]//2-min_w//2:imageB.shape[0]//2+min_w//2,\
        imageB.shape[1]//2-min_w//2:imageB.shape[1]//2+min_w//2]

    imageA_gray = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
    imageA_gray = cv2.resize(imageA_gray,(128,128))
    imageB_gray = cv2.resize(imageB_gray,(128,128))
    # cv2.imwrite("a.jpg",imageA_gray)
    # cv2.imwrite("b.jpg",imageB_gray)

    a = imageA_gray.astype(np.float32)
    b = imageB_gray.astype(np.float32)
    a = (a-127.5)/127.5
    b = (b-127.5)/127.5
    x = np.zeros((1,3,128,128),dtype=np.float32)
    x[0,0,:,:] = a
    x[0,1,:,:] = b
    x = torch.from_numpy(x)
    result = model(x).data.numpy()[0]*128*scale
    print(result)
    # print(result*128)

    imageCorners = np.array([[0,0],[min_w,0],[0,min_w],[min_w,min_w]])
    imagePerturbedCorners = np.array([[0+result[0],0+result[1]],[min_w+result[2],0+result[3]],\
        [0+result[4],min_w+result[5]],[min_w+result[6],min_w+result[7]]])
    H = cv2.getPerspectiveTransform(np.float32(imageCorners), \
        np.float32(imagePerturbedCorners)) 
    print(H)
    result = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    return result

if __name__ == "__main__":
    imageA = cv2.imread("1.jpg")
    imageB = cv2.imread("2.jpg")
    sift_result = SIFT_stitch([imageA,imageB])
    cv2.imwrite("sift_stitch.jpg",sift_result)
    cnn_result = CNN_stitch([imageA,imageB],"checkpoints/108.pkl")
    cv2.imwrite("cnn_sift_stitch.jpg",cnn_result)
