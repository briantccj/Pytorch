import torch
import torchvision
import numpy as np 
import sys
import d2lzh_pytorch as d2l 
import fashion_mnist


batch_size = 256
train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype = torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)

w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition

def net(x):
    return softmax(torch.mm(x.view(-1, num_inputs), w) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

num_epochs, lr = 5, 0.1

d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)

x, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(x[0:9], titles[0:9])
