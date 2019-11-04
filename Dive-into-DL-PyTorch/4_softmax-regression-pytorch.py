import torch
from torch import nn
from torch.nn import init
import numpy as np 
import d2lzh_pytorch as d2l
import fashion_mnist

num_inputs = 28 * 28
num_outputs = 10

batch_size = 256
train_iter, test_iter = fashion_mnist.load_data_fashion_mnist(batch_size)


net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_outputs)
)

init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)