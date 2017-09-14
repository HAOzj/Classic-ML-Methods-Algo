import numpy as np
from random import sample, randint
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DimensionError(ValueError):
    pass


def draw_2d(dataset,k,d,Center,Label):

    colors = cm.rainbow(np.linspace(0, 1, k))
    n = len(dataset)
    if(d == 2):
        Color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        Colors = [Color[i] for i in Label]
        for i in range(n):
            plt.scatter(dataset[i][0], dataset[i][1], color=Colors[i])
        for i in range(k):
            plt.scatter(Center[i][0], Center[i][1],
                        color=Color[i], marker='D')
        plt.show()
    return(Center, Label)

def init_centers(dataset,k,d):
    print("init center")
    n = len(dataset)
    for i in dataset:
        if(len(i) != d):
            raise DimensionError("Data points are not of the given dimension")

    # label stores the clustering of every point
    label = [0 for j in range(n)]
    # choose k distinct samples by random as initial centroids
    center = sample(dataset, k)
    print("initial centers are :", center)
    return center, label


def _k_means(dataset, k, d,Center, Label,maxite=10,):
    n = len(dataset)
    change = 1
    for ite in range(maxite):
        if(change > 0):
            print(ite, "th iteration : ")
            # sum is a list whose element are the sums of points in each cluster in form of array
            Sum1 = [np.array([float(0) for x in range(d)]) for l in range(k)]
            
            # during every iteration, change and dist records the number of changed labels 
            #and the size of each clustering
            change = 0
            Dist = [0 for x in range(k)]
            for j,value in enumerate(dataset):
                # print "let's look at ", j, "th point"
                # kDist is a list storing the distancce between given point and K centers
                KDist = [sum((np.array(value) - np.array(Center[i]))**2) for i in range(k)]
                # dist is the distance between given point and ith center
                clu = KDist.index(min(KDist))
                # print clu
                Dist[clu] += 1
                Sum1[clu] += np.array(dataset[j])
                if(Label[j] != clu):
                    change += 1
                    Label[j] = clu
            # update of centers
            for i in range(k):
                Center[i] = Sum1[i] / float(Dist[i])
            print("new centers are :\n", Center)
            print("new label are : ", Label)

    return Center, Label



def k_means(dataset, k, d, maxite=10):
    """function to perform Lloyd algorithm. If data points are 2-dimentional,
    it will plot all the points colorized according to their cluster
    
    Parameters:
        dataset (Iterable): - a list of data points who are also a list of numbers
        k (int): - number of clusters
        d (int): - dimension of data points
        maxite (int): - maximum of iterations

    Returns:
        List: - list of centers of k clusters, list of labels of n points

    Raise:
        DimensionError: - Data points are not of the given dimension
    """
    Center, Label = init_centers(dataset, k, d)
    Center, Label = _k_means(dataset, k, d,Center, Label)
    draw_2d(dataset,k,d,Center,Label)
    return Center, Label


def main():
    a = [2, 2]
    b = [1, 2]
    c = [1, 1]
    d = [0, 0]
    f = [3, 2]
    dataset = [a, b, c, d, f]
    dataset.append([1.5, 0])
    dataset.append([3, 4])
    res = Kmeans(dataset, 2, 2)
    print(res)

if __name__ == '__main__':
    main()