import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from dataset import dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from models.multiscale_resnet import multiscale_resnet
from utils.train_utils import *

from dataset.data_aug import *


if __name__ == '__main__':
    print("......................................")

    data_pd = dataset.get_image_pd("/usr/Pytorch/imageclassification/dataset/blood-cells/dataset2-master/dataset2-master/images/TRAIN")

    train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=53,stratify=data_pd["label"])

    data_transform = {
        'train': Compose([
            Resize(size=(320, 320)),
            FixRandomRotate(bound='Random'),
            RandomHflip(),
            RandomVflip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            Resize(size=(320, 320)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_set = {}
    data_set["train"] = dataset.dataset(train_pd, data_transform["train"])
    data_set["val"] = dataset.dataset(val_pd, data_transform["val"])

    data_loader = {}
    data_loader["train"] = torch.utils.data.DataLoader(data_set["train"], batch_size=8, shuffle=True, num_workers=2)
    data_loader["val"] = torch.utils.data.DataLoader(data_set["val"], batch_size=8, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = multiscale_resnet(4)
    model = torch.nn.DataParallel(model)
    model.to(device)
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters() , lr = 0.1, momentum=0.9)
    epochs = 10
    best_vaild_loss = float('inf')
    print(best_vaild_loss)

    for epoch in range(epochs):
        
        train_loss, train_acc = train(data_loader["train"], lossfunc, optimizer, model, device)
        val_loss, val_acc = evaluate(data_loader["val"], lossfunc, model, device)

        if val_loss < best_vaild_loss:
            best_vaild_loss = val_loss
            torch.save(model, "./model.pt")

        print('Epoch:{0}|Train Loss:{1}|Train Acc:{2}|Val Loss:{3}|Val Acc:{4}'.format(epoch+1,train_loss,train_acc,val_loss,val_acc))
