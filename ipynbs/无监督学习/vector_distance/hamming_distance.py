import numpy as np

class BinaryArrayError(AttributeError):
    pass

def hamming_distance(x , y):
    """计算汉明距离
    Parameters:
        x (numpy.dnarray): - 向量x,必须为二进制编码
        y (numpy.dnarray): - 向量y,必须为二进制编码
    
    Returns:
        int: - 汉明距离,值域为[0,inf)
    """
    if not (all([isinstance(i,bool) or (isinstance(i,int) and i in (0,1))  for i in x]) and
        all([isinstance(i,bool) or (isinstance(i,int) and i in (0,1))  for i in y])):
        raise BinaryArrayError("数组不是二进制编码数组")
    x = np.array(x)
    y = np.array(y)
    smstr = nonzero(x-y);
    return shape(smstr[0])[1]