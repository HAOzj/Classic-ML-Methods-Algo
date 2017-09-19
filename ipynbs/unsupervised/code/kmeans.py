import numpy as np
from random import sample, randint
from metric import euclidean_distance
from itertools import groupby


def _init_centers(dataset, k):
    """Initilisation of K centres by picking k points in dataset at random 初始化中心点,随机抽取k个点作为中心

    Parameters:
        dataset (Iterable): - sequence of data points 由向量组成的序列
        k (int): - given number of clusters 指定的簇数量
    Returns:
        List: - list of data points after partial labelling in initialization, list of centers 初始化后部分带标签的数据点列表,中心列表
    """
    center_index = sample(range(len(dataset)), k)
    data = [{"data": vec,
             "label": i if i in center_index else None
             } for i, vec in enumerate(dataset)]
    center = [{"data": vec, "label": i}
              for i, vec in enumerate(dataset) if i in center_index]
    return data, center


def _calcul_center(dataset):
    """Caculation of the center of a group of data points 计算一组向量的中心,实际就是计算向量各个维度的均值

    Parameters:
        dataset (Iterable): - list of data points 由向量组成的序列
    Returns:
        List: - center 中心点向量
    """
    n = len(list(dataset))
    return list(sum(np.array(i) for i in dataset) / n)


def _k_means_iter(data, center, last_center=None, count=0, *, maxite=10, distance_func=euclidean_distance, **kws):
    """iterative way of implementing Lloyd k-means的计算迭代器,使用递归的方式编写

    Parameters:
        data (Iterable): - clustered dataset in form of [{"data":xxx,"label":xxx},...]  带标签数据集,格式为[{"data":xxx,"label":xxx},...]
        center (Iterable): - centers in form of [{"data":xxx,"label":xxx},...] 中心位置,格式为[{"data":xxx,"label":xxx},...]
        last_center (Iterable): - centers of last iteration and in form of [{"data":xxx,"label":xxx},...]  上一次的中心位置,格式为[{"data":xxx,"label":xxx},...]
        count (int): - compter of number of iteration 计数器,用于维护迭代次数
        maxite (int): - maximum of iteration 最大迭代次数
        distance_func (Function): - distance function 距离函数
        **kws : - other parameters of distance function 距离函数的其他参数

    Returns:
        List: - clustered/labelled dataset 打好标签的数据集
    """
    if count >= maxite or (
            last_center is not None and sum([distance_func(x["data"], y["data"], **kws) for x, y in zip(center, last_center)]) <= 0.001):
        return data
    else:
        last_center = list(center)
        for i in data:
            min_distance = min(
                [{
                    "distance": distance_func(i["data"], j["data"], **kws),
                    "label":j["label"]
                } for j in center], key=lambda x: x["distance"])
            i["label"] = min_distance["label"]
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
        dataset (Iterable): - sequence of data points who are also a list of numbers
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
