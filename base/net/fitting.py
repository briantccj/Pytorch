import torch
import numpy as np
import torch.utils.data as Data
from matplotlib import pyplot as plt


w, true_b = [1.2, -3.4, 5.6], 5
n_train, n_test = 100, 100

features = torch.randn((n_train + n_test, 1))
features_ploy = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)

lables = w[0] * features_ploy[:, 0] + w[1] * features_ploy[:, 1] + w[2] * features_ploy[:, 2] + true_b

lables += torch.tensor(np.random.normal(0, 0.01, size=len(lables)), dtype=torch.float)



def fit(features_train, lables_train, features_test, lables_test):
    
    batch_size = min(10, len(lables_train))
    dataset = Data.TensorDataset(features_train, lables_train)
    dataiter = Data.DataLoader(dataset, batch_size, shuffle=True)

    print(features_train.shape[-1])

    net = torch.nn.Linear(features_train.shape[-1], 1)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    print(optimizer.values())

    train_ls = []
    test_ls = []

    # epoch_num = 100
    epoch_num = 1

    for _ in range(epoch_num):
        for x, y in dataiter:
            fx = net(x)
            l = loss(fx, y.view(-1, 1)).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


        train_ls.append(loss(net(features_train), lables_train.view(-1, 1)).item())
        test_ls.append(loss(net(features_test), lables_test.view(-1, 1)).item())

    # print(train_ls)
    plt.semilogy(range(1, epoch_num + 1), train_ls)
    plt.semilogy(range(1, epoch_num + 1), test_ls, linestyle=':')
    plt.legend(['train', 'test'])
    # plt.show()


fit(features_ploy[:n_train,:], lables[:n_train], features_ploy[n_train:,:], lables[n_train:])
# fit(features[:n_train,:], lables[:n_train], features[n_train:,:], lables[n_train:])
# fit(features_ploy[:2,:], lables[:2], features_ploy[n_train:,:], lables[n_train:])