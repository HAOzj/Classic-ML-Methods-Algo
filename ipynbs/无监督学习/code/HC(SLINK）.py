'''
Created on the 12th, Sep, 2017

@author : HAO Zhaojun
'''
from math import *


# definition of distance between numeric data points 
# input   : two points a and b,
#           distance used among 'euclidean' 'squared' 'manhattan' and 'max'
# output  : distance between two numeric points
def norm(a,b, metric = 'euclidean'):
	try :
		if(len(a) != len(b)):
			raise ValueError("two vectors are not of the same dimension")
			exit()
		k =0
		for i in range(len(a)):
			if(metric == 'euclidean' or 'squared'):
				k+= (a[i] - b[i])**2
				#print('eucliean',k)
			if(metric =='manhattan'):
				k+= abs(a[i]-b[i])
				#print('manhattan',k)
			if(metric== 'max'):
				k =max(k, abs(a[i]-b[i]))
				#print('max',k)
		if(metric == 'euclidean'):
			k = sqrt(k)
		return(k)
	except TypeError:
		print("Not all data points are numeric")
		
		
# function to execute SLINK algo
# input    : dataset who is a list of data points in form of list,
#			 dimension of data points
# output   : pointer representations of dendrograms Pi and Lambda
def SLINK(Dataset, d):
	n = len(Dataset)
	# All the data points are labelled as 1, 2, ..., n
	# A(i) is Lambda, noting the lowest level at which i is no longer the last point in his cluster  
	# B(i) is the last point in the cluster which i then joins
	A = [10000 for i  in range(n)]
	B = [n*2 for i in range(n)]
	
	#initialisation
	A[0] = 10000
	B[0] = 0
	
	for k in range(1,n):
		B[k] = k
		A[k] = 10000
		M = [0 for i in range(k+1)]
		for i in range(k):
			M[i] = norm(Dataset[i],Dataset[k])
			
		for i in range(k):
			if(A[i]>= M[i]):
				print(i, B[i])
				M[B[i]] = min(M[B[i]], A[i])
				A[i] = M[i]
				B[i] = k
			if(A[i] < M[i]):
				M[B[i]] = min(M[B[i]], M[i])
		for i in range(k):
			if(A[i] >= A[B[i]]):
				B[i] = k 
	return(A,B)
	
## main function
a =[2,2]
b =[1,2]
c =[1,1]
d =[0,0]
e =[3,2]
Dataset = [a,b,c,d,e]
Dataset.append([1.5,0])
Dataset.append([3,4])
res =SLINK(Dataset,2)
print(res)