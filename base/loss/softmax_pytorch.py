import torch
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn

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

class module(nn.Module):
    def __init__(self, input_num, output_num):
        super(module, self).__init__()
        self.liner = nn.Linear(input_num, output_num)

    def forward(self, x):
        return self.liner(x.view(x.shape[0], -1))

net = module(num_input, num_output)
print(net)
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

def train(train_data, test_data, net, loss, optimizer, epochs):

    for epoch in range(epochs):
        
        n = 0
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        for x, y in train_data:
            fx = net(x)
            l = loss(fx, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += len(y)
            train_loss_sum += l.item()
            train_acc_sum += (fx.argmax(dim = 1, keepdim = True).sum().item())
        print("epoch: %d, train: loss %f acc %f" %(epoch, train_loss_sum / n, train_acc_sum / n))

epochs = 5
train(train_iter, test_iter, net, loss, optimizer, epochs)