import torch
import torch.nn as nn
import numpy as np
import math


loss_f = nn.CrossEntropyLoss(weight=None, size_average=True, reduce=False)

#生成网络输出以及目标输出
output = torch.ones(2, 3, requires_grad=True) * 0.5 #假设一个三分类任务，batchsize=2, 假设每个神经元输出都为0.5
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
print(output)
print(target)
loss = loss_f(output, target)

print(loss)

output = output[0].detach().numpy()
output_1 = output[0]
target_1 = target[0].numpy()

print(output)
print(target_1)

x_calss = output[target_1]
print(x_calss)
print(output_1)

exp = math.e
sigma_exp_x = pow(exp, output[0]) + pow(exp, output[1]) + pow(exp, output[2])
log_sigma_exp_x = math.log(sigma_exp_x)

loss_1 = -x_calss + log_sigma_exp_x

print(loss_1)


weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
loss_f = nn.CrossEntropyLoss(weight=weight, size_average=True, reduce=False)
output = torch.ones(2, 3, requires_grad=True) * 0.5 
target = torch.from_numpy(np.array([0, 1])).type(torch.LongTensor)
loss = loss_f(output, target)

print(loss)

loss_f_1 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=1)
loss_f_2 = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False, ignore_index=2)


output = torch.ones(3, 3, requires_grad=True) * 0.5 
target = torch.from_numpy(np.array([0, 1, 2])).type(torch.LongTensor)

loss_1 = loss_f_1(output, target)
loss_2 = loss_f_2(output, target)

print(loss_1)
print(loss_2)

