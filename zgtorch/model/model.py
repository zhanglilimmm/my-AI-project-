import numpy as np
import pickle


class Tensor:
    """
    参数类，把"数值"和"梯度"打包封装在一起，仿PyTorch的Tensor做最必要的简化实现。

    Attributes:
        data: 参数的实际数值(numpy数组)
        grad: 参数对应的梯度(numpy数组，形状和data一致)，backward之前是None
    """

    def __init__(self, data):
        self.data = data
        self.grad = None

    def zero_grad(self):
        """
        把梯度清零。第一次调用时梯度还不存在，先按data的形状分配一份全0数组；
        之后每次调用，直接在原有内存上填0（不重新分配），减少内存开销。
        """
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad.fill(0)


class BaseModel:
    """所有模型的基类"""

    def __init__(self):
        self.w = None      # Tensor类型
        self.b = None      # Tensor类型
        self.input = None  # 缓存本次前向传播的输入，backward时要用它算梯度

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError("子类必须实现forward()方法")

    def parameters(self):
        """
        返回模型的所有可训练参数(一个Tensor对象组成的列表)。
        优化器拿到这个列表后，不需要知道"具体是哪个参数、叫什么名字"，
        就能统一遍历、统一更新——这是让优化器和模型解耦的关键设计。
        """
        return [self.w, self.b]

    def save(self, weight_path):
        weights_dict = {
            "w": self.w.data,
            "b": self.b.data,
        }
        try:
            with open(weight_path, "wb") as f:
                pickle.dump(weights_dict, f)
            print(f"权重已保存到 {weight_path}")
        except Exception as e:
            print(f"保存权重时出错: {str(e)}")

    def load(self, weight_path):
        try:
            with open(weight_path, "rb") as f:
                weights_dict = pickle.load(f)
            self.w.data = weights_dict["w"]
            self.b.data = weights_dict["b"]
            print(f"权重已从 {weight_path} 加载")
        except Exception as e:
            print(f"加载权重时出错: {str(e)}")


class LinearRegression(BaseModel):
    """
    线性回归模型： y = w·x + b

    Args:
        input_dim: 特征维度
    """

    def __init__(self, input_dim):
        super().__init__()
        self.w = Tensor(np.random.randn(input_dim) * 0.01)  # 权重参数 (input_dim,)
        self.b = Tensor(np.array(0.0))                        # 偏置参数 (标量)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.w.data) + self.b.data


class LogicRegression(BaseModel):
    """
    逻辑回归模型： y = sigmoid(w·x + b)，用于二分类

    Args:
        input_dim: 特征维度
    """

    def __init__(self, input_dim):
        super().__init__()
        self.w = Tensor(np.random.randn(input_dim) * 0.01)
        self.b = Tensor(np.array(0.0))

    def forward(self, x):
        self.input = x
        z = np.dot(x, self.w.data) + self.b.data
        z = np.clip(z, -500, 500)  # 防止指数运算溢出
        output = 1 / (1 + np.exp(-z))
        return output
