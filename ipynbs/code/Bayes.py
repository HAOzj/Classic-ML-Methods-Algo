# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 2017

updated on the 20th Sep

@author: Zhaojun
"""


import sys
import numpy as np
import math


def Naive_Bayes(Trainset, dim,last_N={}, last_Y=[]) :
    """Function to execute Naive Bayes Classifier
    Parameters : 
        Trainset(Iterable):- training set with categorical variables at d+1th dimension 
        dim(int):- dimensionality(number of independant variables)
        last_N(Iterable):- acquired list of numbers of occurrences
        last_Y(Iterable):- acquired list of classes
    Return : 
        Dict:- keys are classes and values are lists, whose last element is the number of occurrences of that class and other elements are number of occurrences for each predictor 
    """
    n = len(Trainset)
    last_k = len(last_Y)
    if( set(last_N.keys() ) != set(last_Y)): 
        print('last_N and last_Y are inconsistant')
        exit()

    for i in range(0,n):
        if( len(Trainset[i]) != dim+1 ):
            print("%d th sample is not of %d dimensions"%(i+1, dim))
            exit()
        if(Trainset[i][dim] not in Y):
            Y.append(Trainset[i][dim])
    print("\nY, list of all the modalities : ", Y)
    # k is the number of classes
    k = len(Y)

    for i in range(last_k,k):
        N[Y[i]] =  [{} for d in range(dim)]
        N[Y[i]].append(0)

    for data in Trainset:
        y = data[dim]
        N[ y][dim] += 1
        for d in range(dim):
            if(data[d] not in N[y][d].keys()):
                N[y][d][ data[d] ] =0
            N[y][d][ data[d] ] +=1
    return(N)

def Naive_Predict(X,N,dim):
    """
    Function to predict class based on the acquired Naive Bayes model
    Parameters : 
        X(Iterable):- dataset to predict
        N(Iterable):- acquired Naive Bayes model in form of dictionary 
        dim(int):- dimensionality(number of independant variables)
    Return : 
        probabilities(Iterable):- list of probabilities, probabilities[i][y] represents the probability of ith data belonging to class y
    """
    probabilities = []
    for i in range( len(X)):
        probabilities.append({})
        data = X[i]
        if(len(data) != dim):
            print('%d th data is not of %d dimensions'%(i+1, dim))
            exit()
        for y in N.keys():
            for d in range(dim):
                if(data[d] not in N[y][d].keys()):
                    print('the {0}th variable of {1}th data, {2}, is new to class {3}'.format(d+1,i+1,data[d],y) )
                    probabilities[i][y] = 0
                else :
                    probabilities[i][y] *= N[y][d][ data[d] ]/N[y][dim]
    return(probabilities)


def Gaussian_Bayes( Trainset, dim,  last_N = {}, last_Y=[]) :
    """Function to execute Gaussian Naive Bayes
    Parameters : 

        Trainset(Iterable):- training set whose variables are numerical
        dim(int):- dimensionality(number of independant variables)  
        last_N(Iterable):- acquired list of occurrences
        last_Y(Iterable):- acquired list of classes
    Returns : 
        Dict:- keys are classes and values are lists, whose last element is the number of occurrences of that class and other elements are mu and sigma for each predictor 
    """

    n = len(Trainset)
    last_k = len(last_Y)

    if( set(last_N.keys()) != set(last_Y)): 
        print('last_N and last_Y are inconsistant')
        exit()

    for i in range(0,n):
        if( len(Trainset[i]) != dim+1):
            print("%d th sample is not of %d dimensions"%(i+1, dim))
            exit()
        if(Trainset[i][dim] not in Y):
            Y.append(Trainset[i][dim])
    print("\nY, list of all the modalities : ", Y)
    # k is the number of classes
    k = len(Y)

    for i in range(last_k,k):
        N[Y[i] ] =[ [] for d in range(dim)]

    for data in Trainset:
        for d in range(dim):
            N[ data[dim] ] [d].append(data[d])
    return(N)

def normpdf(x, mu, sigma):
    """
    Function to calculate the probability density function of normal distribution
    Parameters : 
        x(float):- x value
        mu(float):- mu of normal distribution
        sigma(float):- sigma of normal distribution
    Return : 
        float:- probability density function of normal distribution at x point
    """
    if(sigma == 0):
        sigma =float('inf')
    try :
        sigma = float(sigma)
        x = float(x)
        mu = float(mu)
    except ValueError:
        print('x, mu or sigma are not all numeric')
        exit()
    else :
        denom = (2* math.pi)**.5 * sigma
        num = math.exp(-(float(x)-float(mu))**2/(2* sigma**2 ))
        return(num/denom)
	
def Gaussian_Predict(X,N,dim):
    """
    Function to predict class based on the acquired Naive Bayes model
    Parameters : 
        X(Iterable):- dataset to predict
        N(Iterable):- acquired Naive Bayes model in form of dictionary
        dim(int):- dimensionality(number of independant variables)
    Return : 
        probabilities(Iterable):- list of probabilities, probabilities[i][y] represents the probability of ith data belonging to class y
    """

    probabilities = []
    try:
        for y in N.keys():
            for d in range(dim):
                N[y][d] = [np.mean(N[y][d]), np.std(N[y][d]) ]
    except TypeError:
        print('Type of some data in N is wrong')
    except :
        print('Unexpected error:',sys.exc_info()[0])
    else :
        for i in range( len(X)):
            probabilities.append({})
            x = X[i]
            if(len(x) != dim):
                print('%d th data is not of %d dimensions'%(i+1, dim))
                exit()
            for y in N.keys():
                probabilities[i][y] = 1
                for d in range(dim):
                    probabilities[i][y] *= normpdf(x[d], N[y][d][0], N[y][d][1])