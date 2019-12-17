import argparse
import torch
import torch.autograd.function as F
from dataset import H_dataset
import argparse
from network import get_model


def train(args):
    model = get_model(pretrained=True)
    criterion = torch.nn.MSELoss(reduction='sum')
    if(args.optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum)
    elif(args.optimizer == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    train_dataset = H_dataset(args.datasets,train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch,shuffle=True,num_workers=args.num_workers)

    val_dataset =  H_dataset(args.datasets,train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch,shuffle=False,num_workers=args.num_workers)


    gpus = args.gpus
    if(len(gpus)>0):
        model = model.cuda() # It doesn't support multi-card training now.

    for epoch in range(args.epoch):
        train_loss = 0
        for step, (image_pairs, labels) in enumerate(train_loader):
            if(len(gpus)>0):
                image_pairs = image_pairs.cuda()
                labels = labels.cuda()
            pred = model(image_pairs)
            loss = criterion(pred,labels)
            train_loss += loss.data
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if(step%50 == 0):
                print("Epoch: %d | Step: %d | Train Loss: %f "%(epoch,step, train_loss/(step+1)))

        val_loss = 0
        print("Testing----------")
        for step, (image_pairs, labels) in enumerate(val_loader):
            if(len(gpus)>0):
                image_pairs = image_pairs.cuda()
                labels = labels.cuda()
            pred = model(image_pairs)
            loss = criterion(pred,labels)
            val_loss += loss.data
        train_loss = train_loss/(len(train_loader))
        val_loss = val_loss/(len(val_loader))
        print("Epoch: %d | Train Loss: %f | Val Loss: %f "%(epoch, train_loss, val_loss))

     
        torch.save(model.state_dict(),"checkpoints/"+str(epoch)+".pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type = float, default= 1e-4)
    parser.add_argument("--mementum", help="momentum", type = float, default= 0.9 )
    parser.add_argument("--weight_decay", help="weight decay", type = float, default= 1e-4 )
    parser.add_argument("--optimizer", type = str, default="adam", choices=['sgd','adam'])
    parser.add_argument("--epoch", type = int, help="total epoches", default=100)
    parser.add_argument("--batch", type = int, help="batch size",default=128)
    parser.add_argument("--loss",  type = str, help="loss function",default="mse", choices=['mse'])
    parser.add_argument("--datasets",  type = str, help="datasets path",default="datasets/")
    parser.add_argument("--num_workers",type = int, default=8)
    parser.add_argument("--pretrain",type=bool, default= True)
    parser.add_argument("--gpus",type=list, help = "the used gpus",default=[0])
    args = parser.parse_args()
    train(args)






