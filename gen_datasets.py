import cv2
import numpy as np
import random
import argparse 
from tqdm import tqdm
import os

def gen_data(args):
    phase = args.phase
    if(phase == 'train'):
        path = args.train_coco_path
        num = args.train_number
        load_shape = args.train_load_shape
        patch_size = args.train_patch_size
    else:
        path = args.val_coco_path
        num = args.val_number
        load_shape = args.val_load_shape
        patch_size = args.val_patch_size

    image_names = os.listdir(path)
    image_paths = [os.path.join(path,x) for x in image_names]

    image_pairs = []
    labels = []

    for i in tqdm(range(num)):
        image = cv2.imread(image_paths[i],0)                         # Read image

        # cv2.imwrite("raw_image.jpg",image)

        image = cv2.resize(image,(load_shape[0],load_shape[1]))       # Resize image

        # cv2.imwrite("resize_image.jpg",image)
        rho = patch_size/4  # the perturbation range

        for patch_num in range(4):
            #--Random point in image--#
            x = random.randint(rho, load_shape[0] - rho - patch_size)      # not exceed horizontal borders
            y = random.randint(rho, load_shape[1] - rho - patch_size)      # not exceed vertical borders

            #--Draw the square--#
            point1 = (x, y)                                             # top-left
            point2 = (x + patch_size, y)                                # top-right
            point3 = (x, y + patch_size)                                # bottom-left
            point4 = (x + patch_size, y + patch_size)                   # bottom-right
            imageCorners = np.array([point1,point2,point4,point3])

            #--Get patches--#
            raw_patch = image[ y:y + patch_size, x:x + patch_size]                  # Patch of original image

            # cv2.imwrite("raw_patch.jpg",raw_patch)

            randomPerturb = np.random.randint(low=-rho,high=rho,size=(4,2))  # Random values for perturbation
            imagePerturbedCorners = imageCorners + randomPerturb    # Perturb square randomly
            H = cv2.getPerspectiveTransform(np.float32(imageCorners), \
                np.float32(imagePerturbedCorners))                  # Homography, H
            H_inv = np.linalg.inv(H)                                # H^(-1)
        
            imageWarped = cv2.warpPerspective(image, H_inv, (load_shape[0],load_shape[1])) # Warp image using H^(-1)
            # cv2.imwrite("warped_image.jpg",imageWarped)

            warped_patch = imageWarped[ y:y + patch_size, x:x + patch_size]      # Patch of perturbed image
            # cv2.imwrite("warped_patch.jpg",warped_patch)

            #--Features & Labels for network--#
            imageFeature = np.dstack((raw_patch,warped_patch))                 # 2-channel image
            H_4point = imagePerturbedCorners - imageCorners         # 4-point matrix
            image_pairs.append(imageFeature)
            labels.append(H_4point)
    image_pairs = np.stack(image_pairs)
    labels = np.stack(labels)
    np.save("datasets/"+phase+"_data.npy",image_pairs)
    np.save("datasets/"+phase+"_label.npy",labels)

    print(image_pairs.shape)
    print(labels.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",type=str,default="train",choices=['train','val'])
    parser.add_argument("--train_coco_path",type=str,default="/datasets/coco/train2014")
    parser.add_argument("--val_coco_path",type=str,default="/datasets/coco/val2014")
    parser.add_argument("--train_number",type=int,default = 80000)
    parser.add_argument("--val_number",type=int,default = 5000)
    parser.add_argument("--train_load_shape",type=list,default = [320,240])
    parser.add_argument("--val_load_shape",type=list, default=[320,240])
    parser.add_argument("--train_patch_size",type=int, default=128)
    parser.add_argument("--val_patch_size",type=int, default=128)
    args = parser.parse_args()
    gen_data(args)