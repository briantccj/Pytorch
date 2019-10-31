import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


n_train, n_test, feature_num = 20, 100, 200
ture_w, true_b = torch.ones(feature_num) * 0.01, 0.05

features = torch.randn(n_train + n_test, feature_num)
lables = torch.matmul(features, ture_w) + true_b
lables += torch.tensor(np.random.normal(0, 0.01, len(lables)), dtype=torch.float)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_lables, test_labels = lables[:n_train], lables[n_train:]

def l2_penalty(w):
    return (w**2).sum()/2

def init_params():
    w = torch.randn((feature_num, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def linreg(w, b, x):
    return torch.mm(x, w) + b

def squared_loss(fx, y):
    return ((fx - y)**2)/2

def sgd(params, lr, batchsize):
    for param in params:
        param.data -= lr * param.grad/batchsize

batchsize, num_epoch, lr = 1, 100, 0.003
dataset = Data.TensorDataset(train_features, train_lables)
dataiter = Data.DataLoader(dataset, batchsize, shuffle=True)

def train(lambd):    
    w, b = init_params()

    train_ls = []
    test_ls = []
    for _ in range(num_epoch):
        for x, y in dataiter:
            fx = linreg(w, b, x)
            l = squared_loss(fx, y.view(-1, 1)) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            sgd([w, b], lr, batchsize)
        train_ls.append(squared_loss(linreg(w, b, train_features), train_lables.view(-1, 1)).mean().item())
        test_ls.append(squared_loss(linreg(w, b, test_features), test_labels.view(-1, 1)).mean().item())
    print(w.norm().item())
    plt.semilogy(range(num_epoch), train_ls)
    plt.semilogy(range(num_epoch), test_ls)
    plt.show()

# train(0)
# train(1)
# train(3)



def train_pytorch(lambd):

    net = torch.nn.Linear(feature_num, 1)
    loss = torch.nn.MSELoss()
    #对权重参数衰减
    optimizer_w = torch.optim.SGD(params = [net.weight], lr=lr, weight_decay=lambd)
    #不对偏差参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr = lr)

    train_ls = []
    test_ls = []
    for _ in range(num_epoch):
        for x, y in dataiter:
            fx = net(x)
            l = loss(fx, y.view(-1, 1))
            l = l.sum()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(features[:n_train, :]), lables[:n_train].view(-1, 1)).mean().item())
        test_ls.append(loss(net(features[n_train:, :]), lables[n_train:].view(-1, 1)).mean().item())
    print(net.weight.data.norm().item())
    plt.semilogy(range(num_epoch), train_ls)
    plt.semilogy(range(num_epoch), test_ls)
    plt.show()

train_pytorch(0)
train_pytorch(1)
train_pytorch(3)