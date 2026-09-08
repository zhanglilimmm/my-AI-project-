"""
one_hot_test.py

测试 one-hot 编码功能。
one-hot编码常用于多分类任务，把类别标签（0,1,2...）转换成
"只有一位是1、其余都是0"的向量形式，方便神经网络处理。

例如：labels = [0, 2, 1]，num_classes = 3
->  [[1, 0, 0],
     [0, 0, 1],
     [0, 1, 0]]
"""

import numpy as np


def one_hot(labels, num_classes):
    """
    把类别标签转换成one-hot编码矩阵。

    Args:
        labels: 一维数组，每个元素是一个类别的整数编号(从0开始)
        num_classes: 类别总数

    Returns:
        形状为 (n_samples, num_classes) 的one-hot编码矩阵
    """
    labels = np.array(labels).astype(int)
    n_samples = labels.shape[0]
    result = np.zeros((n_samples, num_classes))
    result[np.arange(n_samples), labels] = 1
    return result


if __name__ == "__main__":
    labels = [0, 2, 1, 1, 0]
    num_classes = 3

    encoded = one_hot(labels, num_classes)
    print("原始标签:", labels)
    print("one-hot编码结果:\n", encoded)

    # 简单断言测试，确保编码结果的形状和每一行的和都符合预期
    assert encoded.shape == (len(labels), num_classes), "形状不对"
    assert (encoded.sum(axis=1) == 1).all(), "每一行应该只有一个1"
    for i, label in enumerate(labels):
        assert encoded[i, label] == 1, f"第{i}行对应类别{label}的位置应该是1"

    print("测试通过！")
