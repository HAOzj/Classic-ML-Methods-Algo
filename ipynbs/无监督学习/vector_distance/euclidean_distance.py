import numpy as np


def euclidean_destance(x, y, accuracy=5):
    """计算欧式距离
    Parameters:
        x (numpy.dnarray): - 向量x
        y (numpy.dnarray): - 向量y
    
    Returns:
        float: - 欧式距离,值域为[0,1]
    """
    x = np.array(x)
    y = np.array(y)

    return round(np.linalg.norm(x - y), accuracy)  