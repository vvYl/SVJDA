import numpy as np


def duchi_user_data(v, eps, range_val):
    """
    实现Duchi机制。

    参数:
    v: float，输入值。
    eps: float，隐私预算参数。
    range_val: float，值的范围。
    """
    # 将输入值进行归一化
    v = v / range_val
    e_eps = np.exp(eps)
    c_eps = (e_eps + 1) / (e_eps - 1)

    # 计算概率 p
    p = (v * (e_eps - 1) + e_eps + 1) / (2 * (1 + e_eps))

    # 根据概率 p 生成随机值
    noisy_v = c_eps if np.random.rand() < p else -c_eps

    # 恢复到原范围
    noisy_v *= range_val

    return noisy_v