#基本的张量计算
import torch
import numpy as np
arr=[[2,1],[1,3]]
x=torch.tensor(arr,dtype=torch.float32)
print(x)
print(x.shape)

y=torch.rand(2,3)
print(y)
#这里使用torch.ones生成都为1的张量
x=torch.rand(2,3)
y=torch.zeros_like(x)

#下面是numpy数组和张量的转化，数组逻辑不变
arr_np=np.random.rand(2,3)
x=torch.from_numpy(arr_np)
print(f"numpy array: {arr_np}")
print(f"torch tensor: {x}")

arr_np[0,0]=100
x[0,0]=-100
print(f"after change,numpy array: {arr_np}, torch tensor: {x}")

#以下是基本运算(矩阵运算）

x=torch.tensor([[1,2],[3,4],[5,6]])
y=torch.tensor([[1,1],[2,2],[3,3]])
#method 1
print(f"method 1:{x+y}")
#method 2(减法则为torch.sub(x,y))
print(f"method 2:{torch.add(x,y)}")

#乘除
print(f"multiplication:{torch.mul(x,y)}")
print(f"division:{torch.div(x,y)}")

#张量各个维度计算
x=torch.tensor([[1,2],[3,4],[5,6]],dtype=torch.float32)
print(f"所有元素之和：{x.sum()}")
print(f"沿0维之和：{x.sum(axis=0)}")
print(f"沿1维之和：{x.sum(axis=1)}")

#转置矩阵
print(f"x.T:{x.T}")