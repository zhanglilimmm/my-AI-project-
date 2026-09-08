深度学习框架实现课程作业：使用 NumPy 从零实现模型、Loss 和优化器。
本项目参考 PyTorch 的基本使用流程，使用 NumPy 从底层实现一个简易深度学习框架，实现了模型定义、损失计算、自动求导、参数更新以及完整训练流程。项目主要用于理解深度学习框架底层原理，包括：- 模型结构设计- Loss计算与反向传播- 梯度下降优化算法- 模块化深度学习框架搭建。
一、项目功能
本项目实现了以下核心功能：

## 1. 模型（Model）

目前支持：
### LinearRegression
线性回归模型：

- 输入特征
- 参数 w、b
- 前向预测

### LogicRegression
逻辑回归模型：

- Sigmoid 激活
- 二分类预测

## 2. 损失函数（Loss）

实现：
### MSELoss用于回归任务：
- 计算预测值与真实值之间的均方误差
- 支持反向传播

### CrossEntropyLoss用于分类任务：
- 二分类交叉熵损失
- 支持梯度计算

## 3. 优化器（Optimizer）
实现 7 种梯度下降算法：
- SGD
- Momentum
- Nesterov
- Adagrad
- RMSProp
- Adadelta
- Adam
## 4. 自动求导（Autograd）

实现基础梯度计算机制：

- 保存计算过程中的梯度信息
- 反向传播计算参数梯度
- 支持模型参数更新

训练流程模拟 PyTorch：

```text
optimizer.zero_grad()
        ↓
model(input)
        ↓
loss.backward()
        ↓
optimizer.step()
二、项目结构
zgtorch
│
├── model
│   └── model.py              # LinearRegression、LogicRegression
│
├── loss
│   └── loss.py               # MSELoss、CrossEntropyLoss
│
├── optim
│   └── optimizer.py          # 七种优化器
│
├── data
│   └── dataset.py            # 数据加载
│
├── units
│   └── plot.py               # 可视化工具
│
└── runs
    └── zgtorch_train.py      # 训练入口
三、实验内容
1、线性回归实验
使用身高数据进行训练：
流程：
Dataset
   ↓
LinearRegression
   ↓
MSELoss
   ↓
Adam Optimizer
   ↓
参数更新
训练完成后：输出训练后的权重 w；输出偏置 b；对新输入数据进行预测
示例：
输入：父亲身高 = 90
输出：预测儿子身高
同时绘制：原始数据分布、回归拟合曲线、预测点
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/4bd9add7-fabb-432d-bf4a-026b153ba9fd" />

2、逻辑回归分类实验
使用：Social_Network_Ads.csv进行二分类任务。
训练流程：
Dataset
   ↓
LogicRegression
   ↓
CrossEntropyLoss
   ↓
Adam
输出：模型参数 w、b、分类准确率、分类结果可视化
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/e1157710-aed1-46fd-b2c3-c10a2ea828d7" />

3、七种优化算法对比试验
为了比较不同优化算法的训练效果，使用相同线性回归任务，对以下算法进行 Loss 曲线比较：
实验结果：通过 Loss 曲线可以观察不同优化器在收敛速度和稳定性方面的差异。
<img width="1000" height="600" alt="a77f36214479ca2dd008af9b5efd62bc" src="https://github.com/user-attachments/assets/99902dbb-b553-4c40-a7c5-aff7163084e8" />
## 四、项目特点

- 使用 NumPy 从零实现深度学习训练流程
- 模拟 PyTorch 风格 API
- 理解梯度下降及优化器原理
- 将模型、损失函数、优化器、数据集进行模块化设计
## 五、开发环境
主要依赖：
- numpy
- pandas
- matplotlib
## 六、运行方式
### 1. 安装依赖
pip install numpy pandas matplotlib



