import numpy as np


def minkowski_distance(x , y):
    """计算闵可夫斯基距离
    Parameters:
        x (numpy.dnarray): - 向量x,必须为二进制编码
        y (numpy.dnarray): - 向量y,必须为二进制编码
    
    Returns:
        int: - 闵可夫斯基距离,值域为[0,inf)
    """
    x = np.array(x)
    y = np.array(y)
    
    return np.sqrt(np.sum(np.square(x-y)))