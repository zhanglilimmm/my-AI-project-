"""
zgtorch_train.py

训练入口脚本。调用风格对齐PyTorch，也对齐老师课件里的写法：

    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss_value, loss_fn = loss_fn(output, target)   # 元组解包：数值 + 损失对象本身
        loss_fn.backward()
        optimizer.step()

外层"大for"控制epoch，内层"小for"控制batch。
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from units.plot import LossPlotter, ModelVisualizer
from model.model import LinearRegression, LogicRegression
from loss.loss import MSELoss, CrossEntropyLoss
from optim.optimizer import SGD, Momentum, Nesterov, Adagrad, RMSProp, Adadelta, Adam
from data.classify.dataset import  HeightDataset, CSVDataset, RandomDataset, BaseDataset,TwoSampleDataset

def train(model, loss_fn, optimizer, dataset, epochs=100, print_every=10):
    """
    通用训练函数：大for（epoch） + 小for（batch）
    """
    loss_history = []

    for epoch in range(epochs):              # ------ 大for：控制训练轮数 ------
        total_loss = 0.0
        count = 0

        for input, target in dataset:        # ------ 小for：控制每轮里的batch ------
            optimizer.zero_grad()                          # 1. 清空上一次的梯度
            output = model(input)                          # 2. 前向传播
            loss_value, loss_fn = loss_fn(output, target)   # 3. 计算loss，拿到(数值, 自身)
            loss_fn.backward()                             # 4. 反向传播，把梯度写回model.w.grad/b.grad
            optimizer.step()                               # 5. 用梯度更新参数

            total_loss += loss_value * len(input)
            count += len(input)

        avg_loss = total_loss / count
        loss_history.append(avg_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_history


if __name__ == "__main__":

    # ========================================================
    # Demo 1：线性回归
    # ========================================================
    print("========== 线性回归 (LinearRegression + MSELoss + Adam) ==========")

    height_dataset = HeightDataset(batch_size=4)

    # ---- 4. 创建模型 ----
    height_model = LinearRegression(input_dim=1)

    # ---- 5. 创建损失函数 ----
    height_loss = MSELoss(height_model)

    # ---- 6. 创建优化器 ----
    height_optimizer = Adam(height_model.parameters(), lr=0.01) # 传入 model.parameters()，而不是model本身


    height_history = train( height_model, height_loss, height_optimizer, height_dataset, epochs=500, print_every=20)


    print(f"训练后的权重 w: {height_model.w.data}")
    print(f"训练后的偏置 b: {height_model.b.data}")

    weights_path = os.path.join(os.path.dirname(__file__), "weights.pkl")
    height_model.save(weights_path)
    # ---- 预测儿子身高 ----
    father = np.array([[90]])
    son = height_model(father)
    print( "预测儿子身高:", son )

    visualizer = ModelVisualizer()
    visualizer.regression_plot(height_dataset.X_original,height_dataset.y,height_model,extra_point=(father, son ))




    # ========================================================
    # Demo 2：逻辑回归
    # ========================================================
    print("\n========== 逻辑回归 (LogicRegression + CrossEntropyLoss + Adam) ==========")

    clf_dataset = CSVDataset(     r"E:\workproject\zgtorch_project_2\zgtorch\data\Social_Network_Ads.CSV", batch_size=16 )
    clf_model = LogicRegression(input_dim=2)
    clf_loss_fn = CrossEntropyLoss(clf_model)
    clf_optimizer = Adam(clf_model.parameters(), lr=0.1)

    clf_history = train(clf_model, clf_loss_fn, clf_optimizer, clf_dataset, epochs=200, print_every=20)

    print(f"训练后的权重 w: {clf_model.w.data}")
    print(f"训练后的偏置 b: {clf_model.b.data}")

    import numpy as np
    pred = clf_model(clf_dataset.X)
    pred_label = (pred > 0.5).astype(np.float64)
    acc = (pred_label == clf_dataset.y).mean()
    print(f"训练集准确率: {acc:.2%}")

    visualizer.classification_plot( clf_dataset.X,  clf_dataset.y )

    # ========================================================
    # Demo 3：7种梯度下降算法效果对比（都用线性回归任务测试）
    # ========================================================
    print("\n========== 7种梯度下降算法对比 ==========")

    optimizers_to_test = {
        "SGD": lambda p: SGD(p, lr=0.0001),
        "Momentum": lambda p: Momentum(p, lr=0.0001),
        "Nesterov": lambda p: Nesterov(p, lr=0.0001),
        "Adagrad": lambda p: Adagrad(p, lr=0.05),
        "RMSProp": lambda p: RMSProp(p, lr=0.002),
        "Adadelta": lambda p: Adadelta(p),
        "Adam": lambda p: Adam(p, lr=0.01),
    }
    plotter = LossPlotter()
    loss_results = {}
    for name, opt_builder in optimizers_to_test.items():
        cmp_dataset = HeightDataset(batch_size=2,shuffle=False)
        cmp_model = LinearRegression(input_dim=1)
        cmp_loss_fn = MSELoss(cmp_model)
        cmp_optimizer = opt_builder(cmp_model.parameters())

        history = train(cmp_model, cmp_loss_fn, cmp_optimizer, cmp_dataset,
                         epochs=100, print_every=101)
        loss_results[name] = history
        print(f"{name:10s} 最终Loss: {history[-1]:.6f}")
        print("\n========== 7种梯度下降算法对比 ==========")

        plotter.add(name, history)

# 循环结束以后统一画图
plotter.show()