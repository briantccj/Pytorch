import torch
from IPython import display
from matplotlib import pyplot as plt 
import numpy as np 
import random 
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)).astype(np.float32))

# y = wx + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()).astype(np.float32))

batch_size = 10

dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)

init.normal_(net.linear.weight, mean = 0, std=0.01)
init.constant_(net.linear.bias, val = 0)

loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr = 0.03)

num_epochs = 3

for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print('epoch %d, loss %f' % (epoch, l.item()))

for param in net.parameters():
    print(param)