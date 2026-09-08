import numpy as np


class BaseLoss:
    """
    所有损失函数的基类。

    调用方式仿PyTorch训练脚本里的写法：
        loss_value, loss_fn = loss_fn(output, target)   # __call__返回(数值, 自身)
        loss_fn.backward()                              # 对自身调用backward，把梯度写回模型参数
    """

    def __init__(self, model):
        self.model = model      # 绑定模型，backward时要通过它拿到输入数据、写回梯度
        self.output = None
        self.target = None
        self.n_samples = None

    def __call__(self, output, target):
        self.output = output
        self.target = target
        self.n_samples = output.shape[0] if output.ndim > 0 else 1
        loss_value = self.forward(output, target)
        return loss_value, self  # 元组：(损失数值, 损失函数对象自身)

    def forward(self, output, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _get_input_matrix(self):
        """把模型缓存的input统一reshape成二维矩阵，兼容单样本和批量两种情况"""
        x = self.model.input
        if x.ndim == 1:
            return x.reshape(1, -1)
        return x


class MSELoss(BaseLoss):
    """均方误差损失，用于线性回归： MSE = 1/n * sum((output - target)^2)"""

    def forward(self, output, target):
        return np.mean((output - target) ** 2)

    def backward(self):
        error = self.output - self.target
        input_matrix = self._get_input_matrix()

        grad_w = (2.0 / self.n_samples) * np.dot(input_matrix.T, error)
        grad_b = (2.0 / self.n_samples) * np.sum(error)

        # 梯度写回模型参数(Tensor对象)的.grad属性
        self.model.w.grad = grad_w.reshape(self.model.w.data.shape)
        self.model.b.grad = np.array(grad_b).reshape(self.model.b.data.shape)


class CrossEntropyLoss(BaseLoss):
    """
    二分类交叉熵损失，用于逻辑回归： CE = -1/n * sum(y*log(p) + (1-y)*log(1-p))

    数学性质：当输出是sigmoid(wx+b)时，交叉熵对(wx+b)求导后，
    梯度公式在形式上和MSE一样，都是(output - target)，所以backward写法和MSELoss几乎一致。
    """

    def forward(self, output, target):
        eps = 1e-8  # 防止log(0)出现数值错误
        p = np.clip(output, eps, 1 - eps)
        return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))

    def backward(self):
        error = self.output - self.target
        input_matrix = self._get_input_matrix()

        grad_w = (1.0 / self.n_samples) * np.dot(input_matrix.T, error)
        grad_b = (1.0 / self.n_samples) * np.sum(error)

        self.model.w.grad = grad_w.reshape(self.model.w.data.shape)
        self.model.b.grad = np.array(grad_b).reshape(self.model.b.data.shape)
