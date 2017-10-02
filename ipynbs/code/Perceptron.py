# -*- coding: utf-8 -*-
"""
Created on Mon OCt 02 2017

@author: Zhaojun
"""
from numpy import sign
from numpy import arange 
from matplotlib import pyplot as plt


def dotprod(K,L):
    """dot product of two lists
    Parameters :
        K(Iterable):- a numerical list
        L(Iterable):- a numerical list
    Return ;
        p(float):- dot product of K and L
    """
    if(len(K) == len(L)):
        try:
            return(sum([x*y for x,y in zip(K,L)]) )
        except ValueError:
            print('elements of K and L are not all numeric')
            exit()
    else:
        print("K and L are not of the same length")
		
def _init_weight(Dataset, d, method ="zero"):
    """initialization of weights
    Parameters :
        Dataset(Iterable):- data points with 1 at dth dimension and label at d+1th dimension
        d(int):- dimension of data points
        method(string):- method of initialisation with "zero" by default, "zero" giving zero initial weights while "first" giving the weights the same as the first data point
    Return ;
        w(Iterable):- initial weights
        t(int):- number of updating rounds
    """
    if(method == 'zero'):
        return([0 for i in range(d)],0)
    if(method == 'first' and len(Dataset[0])== d+1):
        return(Dataset[0][:d],0)
def sgn(x):
    """sign of a number, return 1 if the number is non-negative, return -1 otherwise
    Parameters :
        x(float):- a number
    Return ;
        s(int):- 1 if x is non-negative, return -1 otherwise
    """
    if(x>=0):
        s=1
    else:
        s = -1
    return(s)
	
def _update_weights(data, d,last_w,last_t):
    """updating of weights
    Parameters :
        data(Iterable):- labeled data with 1 at dth dimension and label at d+1th dimension
        d(int):- dimension of data points
        last_w(Iterable):-  current weights
        last_t(int):- number of updating rounds
    Return ;
        w(iterable):- weights after this iteration
        t(int):- number of updating rounds after this iteration
    """

    y = dotprod(last_w, data[:d])
    if(sgn(y) != sgn(data[d]) ):
        return( [  x  -sgn(y)*w for x,w in zip(last_w,data[:d])], last_t+1)
        t += 1
    else:
        return(last_w,last_t)

def draw_2d(Dataset,d, w, t):
    """function to plot hyperplane and data points colorized according to their labels if data points are 3-dimentional,
    Parameters:
        Dataset (Iterable): - sequence of data points who are also a list of numbers
        d (int): - dimension of data points
        w(iterable):- weights
        t(int):- number of updating rounds
    """
    n = len(Dataset)
    X = [data[0] for data in Dataset]
    Y = [data[1] for data in Dataset]
    if(d == 3):
        Color = {-1:'b', 1:'g'}
        for i in range(n):
            plt.scatter(Dataset[i][0], Dataset[i][1], color=Color[ Dataset[i][d] ])
        if(w[1] != 0):
            Domain = arange(min(X), max(X), 0.01)
            plt.plot(Domain,[-(w[0]*x + w[2])/w[1] for x in Domain],label = 'hyperplane at '+str(t)+'th update')
            plt.legend()
        else:
            if(w[0] != 0):
                Codomain = arange(min(Y), max(Y), 0.01)
                plt.plot([-(w[1]*y + w[2])/w[0] for y in Codomain],Codomain,label = 'hyperplane at '+str(t)+'th update')
                plt.legend()
        plt.title('%d*x + %d*y+%d =0'%(w[0],w[1],w[2]) )
        plt.savefig('%dth update'%t)
        plt.close()
		
def perceptron(Dataset, d, method ="zero"):
    """implementation of Perceptron algo and save plots of every updating
    Parameters :
        Dataset(Iterable):- labeled data points with 1 at dth dimension and label at d+1th dimension
        d(int):- dimension of data points
        method(string):- method of initialisation with "zero" by default, "zero" giving zero initial weights while "first" giving the weights the same as the first data point
    Return ;
        w(iterable):- final weights
        t(int):- number of updating rounds after this iteration
    """
    w,t = _init_weight(Dataset, d, method)
    draw_2d(Dataset, d, w, t)
    n = len(Dataset)
    for i in range(n):
        if(len(Dataset[i]) != d+1):
            print(i+1,"th data points is not of ", d," dimensions")
        else:
            last_t =t
            last_w = w
            w,t = _update_weights(Dataset[i], d,last_w,last_t)
        if(t != last_t):
            draw_2d(Dataset, d, w, t)
    return(w,t)


		
def main():
    a = [2, 1,1,-1]
    b = [1, 2,1,1]
    c = [1, 3,1,1]
    d = [0, 0.5,1,1]
    f = [3, 2,1,-1]
    Dataset = [a, b, c, d, f]
    Dataset.append([1.5, 0,1,-1])
    Dataset.append([3, 4,1,1])
    res = perceptron(Dataset,3)
    return res


if __name__ == '__main__':
    main()
