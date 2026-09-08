import numpy as np


class BaseOptimizer:
    """
    优化器基类，风格对齐PyTorch：
        optimizer = SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        optimizer.step()

    关键设计：__init__只接收params(一个Tensor列表)，不接收整个model，
    这样优化器完全不需要知道"模型内部有哪些参数、叫什么名字"，
    只要是Tensor对象组成的列表，就能统一处理——不管模型是2个参数还是几百万个。
    """

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        """清空所有参数的梯度"""
        for param in self.params:
            param.zero_grad()

    def step(self):
        raise NotImplementedError


# ============================================================
# 1. SGD —— 最基础的梯度下降： w = w - lr * grad
# ============================================================
class SGD(BaseOptimizer):
    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad


# ============================================================
# 2. Momentum —— 动量法，给梯度更新加上"惯性"
#    v = beta*v + (1-beta)*grad ;  w = w - lr*v
# ============================================================
class Momentum(BaseOptimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * param.grad
            param.data -= self.lr * self.v[i]


# ============================================================
# 3. Nesterov —— Nesterov加速梯度，提前"预判"方向
# ============================================================
class Nesterov(BaseOptimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            v_prev = self.v[i].copy()
            self.v[i] = self.beta * self.v[i] - self.lr * param.grad
            param.data += -self.beta * v_prev + (1 + self.beta) * self.v[i]


# ============================================================
# 4. Adagrad —— 每个参数自适应学习率，梯度累积越多学习率越小
#    s = s + grad^2 ; w = w - lr/sqrt(s+eps) * grad
# ============================================================
class Adagrad(BaseOptimizer):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.s = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.s[i] += param.grad ** 2
            param.data -= self.lr / (np.sqrt(self.s[i]) + self.eps) * param.grad


# ============================================================
# 5. RMSProp —— 改进Adagrad学习率衰减过快的问题，用指数加权平均
#    s = beta*s + (1-beta)*grad^2
# ============================================================
class RMSProp(BaseOptimizer):
    def __init__(self, params, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.s = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * param.grad ** 2
            param.data -= self.lr / (np.sqrt(self.s[i]) + self.eps) * param.grad


# ============================================================
# 6. Adadelta —— RMSProp进阶版，连学习率都不用手动设置
# ============================================================
class Adadelta(BaseOptimizer):
    def __init__(self, params, beta=0.95, eps=1e-6):
        super().__init__(params, lr=None)
        self.beta = beta
        self.eps = eps
        self.s = [np.zeros_like(p.data) for p in self.params]
        self.delta = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            g = param.grad
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * g ** 2
            update = np.sqrt(self.delta[i] + self.eps) / np.sqrt(self.s[i] + self.eps) * g
            param.data -= update
            self.delta[i] = self.beta * self.delta[i] + (1 - self.beta) * update ** 2


# ============================================================
# 7. Adam —— 结合Momentum(一阶矩)和RMSProp(二阶矩)，目前最常用
# ============================================================
class Adam(BaseOptimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            g = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
