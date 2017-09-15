import numpy as np
from random import sample, randint
from vector_distance import euclidean_distance
from itertools import groupby


def _init_centers(dataset, k):
    """初始化中心点,随机抽取k个点作为中心

    Parameters:
        dataset (Iterable): - 由向量组成的序列
        k (int): - 指定的簇数量
    Returns:
        List: - 中心点向量
    """
    center_index = sample(range(len(dataset)), k)
    data = [{"data": vec,
             "label": i if i in center_index else None
             } for i, vec in enumerate(dataset)]
    center = [{"data": vec, "label": i}
              for i, vec in enumerate(dataset) if i in center_index]
    return data, center


def _calcul_center(dataset):
    """计算一组向量的中心,实际就是计算向量各个维度的均值

    Parameters:
        dataset (Iterable): - 由向量组成的序列
    Returns:
        List: - 中心点向量
    """
    n = len(list(dataset))
    return list(sum(np.array(i) for i in dataset) / n)


def _k_means_iter(data, center, last_center=None, count=0, *, maxite=10, distance_func=euclidean_distance, **kws):
    """k-means的计算迭代器,使用递归的方式编写

    Parameters:
        data (Iterable): - 带标签数据集,格式为[{"data":xxx,"label":xxx},...]
        center (Iterable): - 中心位置,格式为[{"data":xxx,"label":xxx},...]
        last_center (Iterable): - 上一次的中心位置,格式为[{"data":xxx,"label":xxx},...]
        count (int): - 计数器,用于维护迭代次数
        maxite (int): - 最大迭代次数
        distance_func (Function): - 计算距离的函数
        **kws : - 距离函数的其他参数

    Returns:
        List: - 打好标签的数据集
    """
    if count >= maxite or (
            last_center is not None and sum([distance_func(x["data"], y["data"], **kws) for x, y in zip(center, last_center)]) <= 0.001):
        return data
    else:
        last_center = list(center)
        for i in data:
            max_distance = max(
                [{
                    "distance": distance_func(i["data"], j["data"], **kws),
                    "label":j["label"]
                } for j in center], key=lambda x: x["distance"])
            i["label"] = max_distance["label"]
        center = []
        gp = groupby(
            sorted(data, key=lambda x: x["label"]), key=lambda x: x["label"])
        for i, v in gp:
            v = list(v)
            temp = _calcul_center([i["data"] for i in v])
            center.append({
                "label": i,
                "data": temp
            })
        return _k_means_iter(data, center, last_center, count=count + 1, maxite=maxite, distance_func=distance_func, **kws)


def k_means(dataset, k, *, maxite=10, distance_func=euclidean_distance, **kws):
    """function to perform Lloyd algorithm. If data points are 2-dimentional,
    it will plot all the points colorized according to their cluster

    Parameters:
        dataset (Iterable): - a list of data points who are also a list of numbers
        k (int): - number of clusters
        d (int): - dimension of data points
        maxite (int): - maximum of iterations

    Returns:
        List: - list of centers of k clusters, list of labels of n points

    """
    data, center = _init_centers(dataset, k)
    return _k_means_iter(data, center, maxite=maxite, distance_func=distance_func, **kws)


def main():
    a = [2, 2]
    b = [1, 2]
    c = [1, 1]
    d = [0, 0]
    f = [3, 2]
    dataset = [a, b, c, d, f]
    dataset.append([1.5, 0])
    dataset.append([3, 4])
    res = k_means(dataset, 2)
    return res


if __name__ == '__main__':
    main()
