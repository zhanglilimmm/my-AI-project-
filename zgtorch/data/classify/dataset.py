import numpy as np
import pandas as pd


class BaseDataset:
    """
    数据集基类，风格参考PyTorch的 Dataset + DataLoader（简化合并成一个类）。

    核心是实现了 __iter__，这样外面才能写：
        for input, target in dataset:
            ...
    每次循环，拿到的是一个batch的数据，而不是单条数据。
    """

    def __init__(self, batch_size=8, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X = None  # 特征，形状 (n_samples, n_features)
        self.y = None  # 标签，形状 (n_samples,)

    def __len__(self):
        """一个epoch里，一共有多少个batch"""
        n_samples = self.X.shape[0]
        return int(np.ceil(n_samples / self.batch_size))

    def __iter__(self):
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)  # 每个epoch开始前，打乱一次顺序

        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


class CSVDataset(BaseDataset):


    def __init__(self, file_path, batch_size=16, shuffle=True):
        super().__init__(batch_size, shuffle)

        # 读取CSV
        data = pd.read_csv(file_path)

        # 分离特征和标签
        self.X = data.iloc[:, :-1].values.astype(np.float64)
        self.y = data.iloc[:, -1].values.astype(np.float64)

        # 原始数据保存，用于画图
        self.X_original = self.X.copy()

        # 标准化（重要！）
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.X = (self.X - self.mean) / self.std

        print(f"✅ 加载数据: {self.X.shape[0]} 个样本, {self.X.shape[1]} 个特征")
        print(f"   标签分布: 0={np.sum(self.y==0)}, 1={np.sum(self.y==1)}")


class HeightDataset(BaseDataset):

    def __init__(self, batch_size=4, shuffle=True):

        super().__init__(batch_size, shuffle)

        self.X_original= np.array(
            [60,62,64,65,66,67,68,70,72,74],
            dtype=np.float64
        ).reshape(-1,1)
        self.X = self.X_original.copy()

        self.y = np.array(
            [63.6,65.2,66,65.5,66.9,
             67.1,67.4,68.3,70.1,70],
            dtype=np.float64
        )

class TwoSampleDataset(BaseDataset):
    def __init__(self,batch_size=2, shuffle=False):
        super().__init__(batch_size=2, shuffle=False)
        self.X = np.array([-1, 1], dtype=np.float64).reshape(-1, 1)
        self.y = np.array([63, 65.2], dtype=np.float64)


class RandomDataset(BaseDataset):
    """
    随机生成一份数据，方便在没有真实数据集时，快速验证模型/优化器能不能跑通。

    Args:
        n_samples: 样本数量
        n_features: 特征维度
        task: "regression"(线性回归) 或 "classification"(逻辑回归二分类)
    """

    def __init__(self, n_samples=200, n_features=3, task="regression",
                 batch_size=8, shuffle=True, seed=42):
        super().__init__(batch_size, shuffle)
        rng = np.random.RandomState(seed)
        self.X = rng.randn(n_samples, n_features)

        true_w = rng.randn(n_features)
        true_b = rng.randn()

        if task == "regression":
            noise = rng.randn(n_samples) * 0.1
            self.y = np.dot(self.X, true_w) + true_b + noise
        elif task == "classification":
            z = np.dot(self.X, true_w) + true_b
            prob = 1 / (1 + np.exp(-z))
            self.y = (prob > 0.5).astype(np.float64)
        else:
            raise ValueError("task 必须是 'regression' 或 'classification'")
