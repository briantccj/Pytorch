import torch
import torch.optim as optim


w1 = torch.randn(2, 2)
w1.required_grad = True

print('w1', w1)

w2 = torch.randn(2, 2)
w2.required_grad = True

print('w1', w1)

optimizer = optim.SGD([w1, w2], lr = 0.001, momentum = 0.9)
optimizer.param_groups[0]['params'][0].grad = torch.randn(2, 2)

print('参数w1的梯度:')
print(optimizer.param_groups[0]['params'][0].grad) #参数组第一个参数w1的梯度

optimizer.zero_grad()
print('执行zero_grad()之后，参数w1的梯度：')
print(optimizer.param_groups[0]['params'][0].grad)