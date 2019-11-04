import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import time
import sys
import d2lzh_pytorch as d2l


def load_data_fashion_mnist(batch_size):

    mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))

    feature, label = mnist_train[0]
    print(feature.shape, label)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # start = time.time()
    # for X, y in train_iter:
    #     continue
    # print('%.2f sec' %(time.time() - start))

    return train_iter, test_iter
