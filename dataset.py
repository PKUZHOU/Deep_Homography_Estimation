import torch
import torch.nn as nn
import numpy as np
import os
class H_dataset(torch.utils.data.Dataset):
    def __init__(self,data_path, train =True, transform=None,norm = True):
        self.data_path = data_path
        self.norm = norm
        if(train):
            data = os.path.join(self.data_path,"train_data.npy")
            label = os.path.join(self.data_path,"train_label.npy")
        else:
            data = os.path.join(self.data_path,"val_data.npy")
            label = os.path.join(self.data_path,"val_label.npy")
        self.data = np.load(data)
        self.label = np.load(label)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        image_pairs = self.data[index]
        image_pairs = image_pairs.transpose(2,0,1)
        if(self.norm):
            image_pairs = (image_pairs - 127.5)/127.5
            # add an extra input channel, because I want to use the pretrained model on imagenet.
            image_pairs_3c = np.zeros((3,128,128),dtype=np.float32) 
            image_pairs_3c[:2,:,:] = image_pairs
            image_pairs = image_pairs_3c
        label = self.label[index]
        if(self.norm):
            label = label.reshape(-1).astype(np.float32)
        label = label/128.
        return [image_pairs,label]

if __name__ == "__main__":
    data = H_dataset("./datasets")
    print(len(data))
    print(data[0])