# coding: utf-8
import torch
from PIL import Image
from torch.utils.data import Dataset

##第一步继承Dataset定义自己的数据集
##主要是实现__getitem__(self, index)和__len__(self)
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs 
        self.transform = transform
        self.target_transform = target_transform

    #得到数据内容和标签
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
        return img, label

    #返回数据集大小    
    def __len__(self):
        return len(self.imgs)

##第二步以自定义数据集为参数定义一个DataLoader类，这个类就是可以在for循环中使用的可迭代对象
mydataset = MyDataset('data')
train_iter = torch.utils.data.DataLoader(mydataset, batch_size=64, shuffle=True, num_workers=0)

for epoch in range(10):
    for x, y in train_iter:
        print(x)
# 总结：
# 1.Dataset是一个抽象类，需要派生一个子类构造数据集，需要改写的方法有__init__，__getitem__等。
# 2.DataLoader是一个迭代器，方便我们访问Dataset里的对象，
#   值得注意的num_workers的参数设置：如果放在cpu上跑，可以不管，但是放在GPU上则需要设置为0；
