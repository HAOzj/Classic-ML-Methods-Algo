import numpy as np


def chebyshev_distance(x, y, accuracy=5):
    """计算切比雪夫距离
    Parameters:
        x (numpy.dnarray): - 向量x
        y (numpy.dnarray): - 向量y
    
    Returns:
        float: - 切比雪夫距离,值域为[0,inf)
    """
    x = np.array(x)
    y = np.array(y)
    return round(max(abs(x - y)), accuracy)