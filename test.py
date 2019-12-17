import argparse
import torch
import torch.autograd.function as F
from dataset import H_dataset
import argparse
from network import get_model
from sift_ransac import sift_ransac
import numpy as np
import cv2

def metric(pred, label):
    pred = pred.reshape(-1,2)
    label = label.reshape(-1,2)
    pred = pred*128
    label = label*128
    # print(pred)
    # print(label)
    error = np.sum(np.sqrt((pred[:,0]-label[:,0])**2+(pred[:,1]-label[:,1])**2).reshape(-1))/4
    return error

def H_transform(points,H):
    points = points.reshape(-1,4,2)
    # expanded = np.ones((points.shape[0],3),dtype=np.float32)
    # expanded[:,:2] = points
    # print(expanded.T)
    # transformed = np.matmul(H,expanded.T)
    # print(transformed)
    # transformed = transformed.T[:,:2].reshape(-1,4,2)
    # transformed[:,1,0]-=128
    # transformed[:,2,1]-=128
    # transformed[:,3,:]-=128
    # transformed = transformed.reshape(-1,2)/128.
    transformed = cv2.perspectiveTransform(points, H)
    transformed[:,1,0] = transformed[:,1,0]-128
    transformed[:,2,1] = transformed[:,2,1]-128
    transformed[:,3,:] = transformed[:,3,:]-128
    transformed = -transformed.reshape(-1,2)/128.
    return transformed

def test(args):
    if(args.mode == 'cnn'):
        val_dataset =  H_dataset(args.datasets,train=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch,shuffle=False,num_workers=args.num_workers)
        model = get_model(pretrained=True)
        model.load_state_dict(torch.load(args.model))
        model.cuda()
        val_loss = 0
        # print(len(val_loader))
        for step, (image_pairs, labels) in enumerate(val_loader):
            if(len(args.gpus)>0):
                image_pairs = image_pairs.cuda()
                labels = labels.cuda()
            pred = model(image_pairs)
            loss = metric(pred.cpu().data.numpy(),labels.cpu().data.numpy())
            val_loss += loss
        val_loss = val_loss/(len(val_loader))
        print("CNN average corner error: ",val_loss)
    else:
        val_dataset =  H_dataset(args.datasets,train=False,norm=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch,shuffle=False,num_workers=args.num_workers)
        success = 0
        val_loss = 0
        for step, (image_pairs, labels) in enumerate(val_loader):
            try:
                H = sift_ransac(image_pairs[0].cpu().numpy())
                
                if(H is not None):
                    success+=1
                    corners = np.zeros((image_pairs.size(0),4,2),dtype = np.float32)
                    corners[:,0,:] = 0
                    corners[:,1,0] = 128
                    corners[:,1,1] = 0
                    corners[:,2,0] = 0
                    corners[:,2,1] = 128
                    corners[:,3,:] = 128
                    corners = corners.reshape(-1,2)
                    pred = H_transform(corners,H)
                    loss = metric(pred,labels.cpu().data.numpy())
                    val_loss += loss
                    # print(loss)
            except:
                pass
                # break
        val_loss = val_loss/success
        print("SIFT average corner error: ",val_loss)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = int, help="batch size",default=1)
    parser.add_argument("--datasets",  type = str, help="datasets path",default="datasets/")
    parser.add_argument("--model",  type = str, help="trained model",default="checkpoints/108.pkl")
    parser.add_argument("--num_workers",type = int, default=8)
    parser.add_argument("--mode",type = str, default="sift",choices=['sift','cnn'])
    parser.add_argument("--gpus",type=list, help = "the used gpus",default=[0])
    args = parser.parse_args()
    test(args)


