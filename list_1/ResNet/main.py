import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from torch.distributions import MultivariateNormal
from torch import optim

# 设置两个高斯分布的均值向量和协方差矩阵
mu1=-3*torch.ones(2)
mu2=3*torch.ones(2)
sigma1=torch.eye(2)*0.5
sigma2=torch.eye(2)*2

#从两个多元高斯分布中生成100个样本
m1=MultivariateNormal(mu1,sigma1)
m2=MultivariateNormal(mu2,sigma2)
x1=m1.sample((100,))
x2=m2.sample((100,))

#设置正负样本的标签
y =torch.zeros((200,1))
y[100:]=1

#组合、打乱样本
x=torch.cat([x1,x2],dim=0)
idx=np.random.permutation(len(x))
x=x[idx]
y=y[idx]

#绘制样本
plt.scatter(x1.numpy()[:,0],x1.numpy()[:,1])
plt.scatter(x2.numpy()[:,0],x2.numpy()[:,1])
plt.show()

#设置变量维度
D_in,D_out=2,1
linear=nn.Linear(D_in,D_out,bias=True)
output=linear(x)
# print(x.shape,linear.weight.shape,linear.bias.shape,output.shape)


#定义模型
class LogisticRegression(nn.Module):
    def __init__(self,D_in):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(D_in,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.linear(x))

#调用函数计算二值交叉熵损失
lr_model = LogisticRegression(2)
loss=nn.BCELoss()
#构建优化器，传入待学习参数及学习率
optimizer = optim.SGD(lr_model.parameters(),lr=0.03)

batch_size=200
iters=10
#迭代对模型进行训练
for _ in range(iters):
    for i in range(int(len(x)/batch_size)):
        input = x[i*batch_size:(i+1)*batch_size]  #选取样本
        target = y[i*batch_size:(i+1)*batch_size]  #选取标签
        optimizer.zero_grad()  #清空参数的梯度
        output = lr_model(input)  #调用模型
        l = loss(output,target)  #损失值
        l.backward()  #计算模型梯度
        optimizer.step()  #更新模型的参数


preg_neg = (output <=0.5).view(-1)  #小于等于0.5设为负类
preg_pos = (output >0.5).view(-1)   #大于0.5设为正类
plt.scatter(x[preg_neg,0],x[preg_neg,1])  #画出预测点图
plt.scatter(x[preg_pos,0],x[preg_pos,1])
w = lr_model.linear.weight[0]  #接收模型参数
b = lr_model.linear.bias[0]   #接收模型参数

#画出边界
def draw_decision_boundary(w,b,x0):
    x1 = (-b-w[0]*x0)/w[1]
    plt.plot(x0.detach().numpy(),x1.detach().numpy(),'r')
    plt.show()

#调用画图函数
draw_decision_boundary(w,b,torch.linspace(x.min(),x.max(),50))