import torch
import torch.optim as optim

w1 = torch.randn(2, 2)
w1.requires_grad = True

w2 = torch.randn(2, 2)
w2.requires_grad = True

w3 = torch.randn(2, 2)
w3.requires_grad = True

optimizer_1 = optim.SGD([w1, w2], lr = 0.1)
print('当前参数组个数：', len(optimizer_1.param_groups))
print(optimizer_1.param_groups)

print('增加一个参数 w3')
optimizer_1.add_param_group({'params': w3, 'lr': 0.001, 'momentum':0.8})

print('当前参数组个数：', len(optimizer_1.param_groups))
print(optimizer_1.param_groups)

print('可以看到，参数组是一个list， 一个元素是一个dict，每个dict中有lr， momentum等参数，这些都是可单独管理，单独设定，十分灵活！')