import numpy as np


def cos_distance(x, y, accuracy=5):
    """计算余弦距离
    Parameters:
        x (numpy.dnarray): - 向量x
        y (numpy.dnarray): - 向量y
    
    Returns:
        float: - 余弦距离,值域为[0,1]
    """
    x = np.array(x)
    y = np.array(y)
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    return round((x.dot(y))/(Lx*Ly),accuracy)