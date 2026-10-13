import torch
import torch.optim as optim

# ① 最基础的自动求导
x = torch.tensor(2., requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(f"dy/dx = {x.grad}")        # 数学上 = 2x+3 = 7

# ② 为什么必须 zero_grad —— 梯度会累加
x = torch.tensor(2., requires_grad=True)
for i in range(3):
    y = x**2
    y.backward()
    print(f"第{i+1}次 backward: x.grad = {x.grad}")   # 4 → 8 → 12
#如果使用梯度清零
for i in range(3):
    x.grad.zero_()
    y = x**2
    y.backward()
    print(f"第{i+1}次 backward: x.grad = {x.grad}")

# ③ backward() 只能对标量
x = torch.randn(3, requires_grad=True)
y = x**2
try:
    y.backward()
except RuntimeError as e:
    print(f"报错: {e}")
y.sum().backward()                 # 正确做法：先聚合成标量
print(f"求和后: {x.grad}")

# ④ no_grad 的作用
with torch.no_grad():
    z = x * 2
print(f"no_grad 内的张量 requires_grad = {z.requires_grad}")   # False

#重要用法：view函数，便捷地改变张量的维度和大小
x=torch.rand(4,4,dtype=torch.float32)
y=x.view(16)
print(y)
z=y.view(-1,2)#-1从其他维度判断该维度元素数
print(z)
#对于只有一个元素的张量可以用.item将其取出
x=torch.tensor(1.,dtype=torch.float32)
print(x.item())

#接下来是深度学习中常用的GPU设备判断
if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device)#直接在GPU上创建
    x=x.to(device=device)#将张量放入GPU
    z=x+y
    print(z)
