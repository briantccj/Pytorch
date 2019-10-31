import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import time
import sys

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
    
    return train_iter, test_iter

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_input = 28 * 28
num_output = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_input, num_output)), dtype=torch.float32)
b = torch.zeros(num_output, dtype=torch.float32)

w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

def softmax(y):
    y_exp = y.exp()
    return y_exp / y_exp.sum(dim = 1, keepdim=True)

def net(x):
    return softmax(torch.mm(x.view(-1, num_input), w) + b)

def cross_entropy(fx, y):
    return -torch.log(fx.gather(1, y.view(-1, 1)))

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

def evaluate_accuracy(iterator, net):
    
    n = 0
    acc_sum = 0.0
    for x, y in iterator:
        fx = net(x)
        acc_sum += (fx.argmax(dim = 1) == y).float().sum().item()
        n += len(y)
    return acc_sum / n


def train(train_iterator, eval_iterator,net, loss, epochs, params, lr, batch_size):

    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        for x, y in train_iterator:
            fx = net(x)
            l = loss(fx, y).sum()

            if params[0].grad is not None :
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            
            sgd(params, lr, batch_size)

            train_loss_sum += l.item()
            train_acc_sum += (fx.argmax(dim = 1) == y).float().sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(eval_iterator, net)
        print("train: loss  %f acc %f, test: acc %f" 
                % (train_loss_sum / n, train_acc_sum / n, test_acc))


train(train_iter, test_iter, net, cross_entropy, 50, [w, b], 0.1, batch_size)
