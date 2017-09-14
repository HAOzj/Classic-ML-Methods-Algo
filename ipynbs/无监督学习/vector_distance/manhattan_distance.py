import numpy as np


def manhattan_distance(x , y):
    """计算曼哈顿距离
    Parameters:
        x (numpy.dnarray): - 向量x,必须为二进制编码
        y (numpy.dnarray): - 向量y,必须为二进制编码
    
    Returns:
        int: - 哈顿距离,值域为[0,inf)
    """
    x = np.array(x)
    y = np.array(y)
    return sum(abs(x-y))