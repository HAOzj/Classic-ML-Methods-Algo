'''
Created on the 12th Sep, 2017

Updated on the 18th Sep, 2017
@author : HAO Zhaojun
'''
from math import *
inf = float('inf')


def metrics(a, b, met='euclidean'):
    """definition of distance between numeric data points
    parameters :
        a,b(Iterator)  : - two points to compare
        metric(string) : - used metric, "euclidean", "squared", "manhattan" or "max"
    Return  :
        distance(float) : - distance between two numeric points
    """
    try:
        if(len(a) != len(b)):
            raise ValueError("two vectors are not of the same dimension")
            exit()
        k = 0
        for i in range(len(a)):
            if(met == 'euclidean' or 'squared'):
                k += (a[i] - b[i])**2
            if(met == 'manhattan'):
                k += abs(a[i] - b[i])

            if(met == 'max'):
                k = max(k, abs(a[i] - b[i]))

        if(met == 'euclidean'):
            k = sqrt(k)
        return(k)
    except TypeError:
        print("Not all data points are numeric")


def _init():
    """Initialization of pointer representation

    Returns:
        List: - list of pointer representation of the dendrogram of the first element
    """
    A = [inf]
    B = [0]
    return(A, B)


def _SLINK_recur(Dataset, A=None, B=None, dist=metrics, **kws):
    """recursion of SLINK

    Parameters:
        Dataset (Iterable): - dataset
        A(Iterable), B(Iterable) : - pointer representations of the dendrogram of the first k elements
        dist(Function) :- used metric
        **kws : - other parameters of metric function
    Returns :
        List: - pointer representations of dendrogram of the first k+1 elements with A noting the lowest level at which i is no longer the last point in his cluster and B storing the last point in the cluster which i then joins


    Returns:
        List: -
    """
    if(A is None and B is None):
        k = 1
        A = [inf]
        B = [0]
    else:
        if(len(A) == len(B)):
            k = len(A)
        else:
            print('A and B are not of same dimension')
            exit()
    n = len(Dataset)
    if(k >= n):
        return(A, B)
    else:
        B.append(k)
        A.append(inf)
        M = [0 for i in range(k)]
        for i in range(k):
            M[i] = dist(Dataset[i], Dataset[k])

        for i in range(k):
            if(A[i] >= M[i]):
                M[B[i]] = min(M[B[i]], A[i])
                A[i] = M[i]
                B[i] = k
            if(A[i] < M[i]):
                M[B[i]] = min(M[B[i]], M[i])
        for i in range(k):
            if(A[i] >= A[B[i]]):
                B[i] = k
        return _SLINK_recur(Dataset, A, B)


# def SLINK_recur(Dataset, A=None, B=None, dist=metrics, **kws):
#     """function to execute SLINK algo in recursive way
#     Parameters :
#         Dataset(List) : - list of data points, who are also lists
#         A(Iterable), B(Iterable) : - pointer representations of the dendrogram of the first k elements
#         dist(Function) :- used metric
#         **kws : - other parameters of metric function
#     Returns :
#         List: - pointer representations of dendrograms with A noting the lowest level at which i is no longer the last point in his cluster and B storing the last point in the cluster which i then joins
#     """
#     if(A == None and B == None):
#         A, B = _init()
#     return _SLINK_recur(Dataset, A, B, dist=metrics, **kws)


def SLINK(Dataset, d):
    """function to execute SLINK algo


    Parameters :
        Dataset(List) : - list of data points, who are also lists
        d(int) : - dimension of data points


    Returns :
        res(Iterables) : - list of triples sorted by the second element,\
        first element is index of point,\
        the other two are pointer representations of dendrograms noting the\
        lowest level at which i is no longer the last point in his cluster and\
        the last point in the cluster which i then joins\

        Heights(Iterables) : - list of the second element of res' triples
    """
    n = len(Dataset)
    A = [inf for i in range(n)]
    B = [0 for i in range(n)]

    # initialisation
    A[0] = inf
    B[0] = 0

    for k in range(1, n):
        B[k] = k
        A[k] = inf
        M = [0 for i in range(k + 1)]
        for i in range(k):
            M[i] = metrics(Dataset[i], Dataset[k])
        for i in range(k):
            if(A[i] >= M[i]):
                M[B[i]] = min(M[B[i]], A[i])
                A[i] = M[i]
                B[i] = k
            if(A[i] < M[i]):
                M[B[i]] = min(M[B[i]], M[i])
        for i in range(k):
            if(A[i] >= A[B[i]]):
                B[i] = k
    res = [(index, i, j) for index, (i, j) in enumerate(zip(A, B))]
    res = sorted(res, key=lambda x: x[1])
    Heights = [triple[1] for triple in res]
    return(res, Heights)


def display_from_SLINK(res, Heights):
    '''function to display clustering layer by layer

    according to the pointer representations of dendrograms

    Parameters :
        res(Iterables) : - list of triples sorted by the second element,\
            first element is index of point,\
            the other two are pointer representations of dendrograms noting the\
            lowest level at which i is no longer the last point in his cluster and\
            the last point in the cluster which i then joins

        Heights(Iterables) : - list of the second element of res' triples
    '''
    num_points = len(Heights)
    Clustering = [[i] for i in range(num_points)]
    i = 0
    j = 0
    k = 0
    while i < num_points - 1:
        # one layer clustering
        i = j
        k += 1
        print(k, 'th 合并')
        while j < num_points - 1 and Heights[j] == Heights[i]:
            Clustering[res[j][2]] += Clustering[res[j][0]]
            print(Clustering[res[j][2]])
            j += 1
