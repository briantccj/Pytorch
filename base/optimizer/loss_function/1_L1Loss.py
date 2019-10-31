import torch
import torch.nn as nn

output = torch.ones(2, 2, requires_grad=True) * 0.2
target = torch.ones(2, 2)

reduce_false = nn.L1Loss(size_average = True, reduce = False)
size_average_true = nn.L1Loss(size_average = True, reduce = True)
size_average_false = nn.L1Loss(size_average = False, reduce = True)

o_0 = reduce_false(output, target)
o_1 = size_average_true(output, target)
o_2 = size_average_false(output, target)


print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))
print('size_average=True，\t求平均:\t{}'.format(o_1))
print('size_average=False，\t求和:\t{}'.format(o_2))


# print(o_0)
# print(o_1)
# print(o_2)
# print(format(o_0))
# print(format(o_1))
# print(format(o_2))