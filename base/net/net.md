模型定义的三个步骤
1.继承类nn.Module
2.在__init__(self)中定义需要的层（如conv，pooling，relu等）
3.在forward(self, x)中连接各层




nn.Sequetial

torch.nn.Sequetial就是Sequential容器，该容器将一系列操作按照先后顺序给包起来，方便使用。



