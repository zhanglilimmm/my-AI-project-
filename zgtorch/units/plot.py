import matplotlib.pyplot as plt


class LossPlotter:
    """
    Loss曲线绘制工具
    """

    def __init__(self):
        self.loss_results = {}


    def add(self, name, history):
        """
        添加一个优化器的loss记录

        name:
            优化器名字

        history:
            train函数返回的loss_history
        """
        self.loss_results[name] = history


    def show(self, title="Optimizer Comparison"):
        """
        绘制loss曲线
        """

        plt.figure(figsize=(10, 6))

        for name, history in self.loss_results.items():
            plt.plot(history, label=name)


        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(title)

        plt.legend()

        plt.grid(True)

        plt.show()

class ModelVisualizer:
    """
    模型结果可视化工具
    """

    def regression_plot(self, X, y, model, extra_point=None):
        """
        线性回归: 真实值 + 拟合直线
        """
        pred = model(X)

        plt.figure(figsize=(8, 5))

        plt.scatter(
            X,
            y,
            label="True"
        )

        plt.plot(
            X,
            pred,
            label="Prediction"
        )

        # 新预测点
        if extra_point is not None:
            x_new, y_new = extra_point

            plt.scatter(
                x_new,
                y_new,
                marker="*",
                s=200,
                label="New Prediction"
            )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression Result")

        plt.legend()
        plt.grid()

        plt.show()

    def classification_plot(self, X, y):
        """
        逻辑回归分类结果
        """
        plt.figure(figsize=(8, 5))

        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y
        )

        plt.xlabel("Age")
        plt.ylabel("Salary")

        plt.title("Classification Result")

        plt.grid()

        plt.show()