# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 2017
@author: Zhaojun
"""


import sys
import numpy as np
import math

	
# function to execute Naive Bayesian, a basic and widely used classifier
# input : training set with categorical  variables, dimensionality(number of independant variables),  sample to predict, 
#			path of an output file with null by default 
# output : output file showing metadonnes, predicted value
def Bayesian_discrete(trainset, dim,  x, f = 'nul') :
	sys.stdout = open(f,'w')
	print("#################################\n","the sample to predict is : ", x, sep="")
	n = len(trainset)
	# N is the matrice of conditional probability
	N = {}
	# Y is the list of all classes
	Y = []
	# Ny is the vector of distribution in different classes
	Ny = {}
	
	for i in range(0,n):
		if(trainset[i][dim] not in Y):
			Y.append(trainset[i][dim])
	print("\nY, list of all the modalities : ", Y,sep="")
	modality = len(Y)

	for y in Y :
		Ny[y] = 0
	
	for j in range(0,dim):
		N[x[j]] = {}
		for k in range(0,modality):
			N[x[j]][Y[k]] = 0

	for i in range(0,n):
		k = Y.index(trainset[i][dim])
		for j in range(0,dim):
			if(trainset[i][j] == x[j]):
				N[x[j]][Y[k]] += 1
		Ny[Y[k]] += 1
	print("x:{y : nmb} indicates that the number of samples taking value x and belonging to Y[y] class is nmb : \n", N)
	print("number of samples in different classes are : ",Ny)
	probability = [Ny[Y[k]] for k in range(0,modality)]
	
	print("\nIn this case,")
	for k in range(0,modality):
		print("given %s : "%Y[k])
		for j in range(0,dim):
			print("    the conditional probabilities {3} are : {0} / {1}  = {2} ".format(N[x[j]][Y[k]],Ny[Y[k]] , float(N[x[j]][Y[k]])/Ny[Y[k]], x[j], Y[k] ) )
			probability[k] *= float(N[x[j]][Y[k]])/Ny[Y[k]]
			
	b = probability.index(max(probability))
	print("The prediction is %s" %Y[b])
	return Y, Y[b], probability
	
# function to execute Gaussian Naive Bayes
# input : training set whose variables are numerical, dimensionality(number of independant variables),  
#			sample to predict, path of an output file with null by default 
# output : output file showing metadonnes, predicted value
def Bayesian_Gaussian(trainset, dim,  x, f = 'nul') :
	sys.stdout = open(f,'w')
	print("#################################\n","the sample to predict is : ", x, sep="")
	n = len(trainset)
	# y is the list of all classes
	Y = []
	# Ny is the vector of distribution in different classes
	Ny = {}
	N = {}
	Para = {}

	for i in range(0,n):
		if(trainset[i][dim] not in Y and len(trainset[i][dim]) >0):
			Y.append(trainset[i][dim])
			print("the new class is found in %d th line"%i)
	print("\nY, list of all the modalities : ", Y,sep="")
	modality = len(Y)

	for y in Y :
		Ny[y] = 0
	
	for j in range(0,dim):
		Para[j] = {}
		N[j] = {}
		for k in Y:
			N[j][k] = []
			Para[j][k] = []

	for i in range(0,n):
		k = trainset[i][dim]
		for j in range(0, dim):
			try :
				N[j][k].append(trainset[i][j])
			except :
				print(i, j , k)
		Ny[k] += 1
	print("number of samples in different classes are : ",Ny)
	
	for j in range(0,dim):
		for k in Y:
			mea = np.mean(N[j][k])
			var = np.var(N[j][k])
			print("Given class {2}, mean of {3}th variable is {0}, variance is {1}".format(mea, var, k, j+1))
			Para[j][k].append(mea)
			Para[j][k].append(var)
			
	probability = [Ny[Y[k]] for k in range(0,modality)]
	
	print("\nIn this case,")
	for y in Y:
		print("given %s : "%y)
		for j in range(0,dim):
			gau = normpdf(x[j],Para[j][y][0],Para[j][y][1])
			print("    the conditional probabilities {3} are : {0} ".format(gau,Ny[y] , float(gau)/Ny[y], x[j]) )
			probability[Y.index(y)] *= float(gau)
			
	b = probability.index(max(probability))
	print("The prediction is %s" %Y[b])
	return Y, Y[b], probability