'''
Created on the 11th, Sep, 2017
@author : HAO Zhaojun
'''

from numpy import *
from random import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# function to perform Lloyd algorithm. If data points are 2-dimentional, it will plot all the points colorized according to their cluster
# input   : Dataset a list of data points who are also a list of numbers,
#           k number of clusters
#			d dimension of data points
#			maxite maximum of iterations
# output  : list of centers of k clusters, list of labels of n points 
def Kmeans(Dataset, k, d, maxite = 10) :
	# dataset is a list whose elements are also list
	n = len(Dataset)
	Colors = cm.rainbow(linspace(0, 1, k))
	try :
		for i in Dataset:
			if(len(i) != d):
				raise ValueError("Data points are not of the given dimension")
				exit()

		# label stores the clustering of every point
		Label = []
		for j in range(n):
			Label.append(0)
		# choose k distinct samples by random as initial centroids
		nomb = 1
		Center=[]
		num = randint(0,n-1)
		print(num)
		Center.append(Dataset[num])
		while(nomb < k) :
			num = randint(0,n-1)
			print(num)
			rep = 0
			for i in range(0, nomb) :
				if((Dataset[num] > Center[i]) - (Dataset[num] < Center[i])==0):
					rep+= 1
			if(rep ==0):
				Center.append(Dataset[num])
				nomb += 1
			
		print("initial centers are :", Center)
		
		change = 1

		for ite in range(maxite):
			if(change>0):
				print(ite, "th iteration : ")           
				# sum is a list whose element are the sums of points in each cluster in form of array
				Sum1 = []
				for l in range(k):
					Sum1.append(array([float(0) for x in range(d)]))
				# during every iteration, change and dist records the number of changed labels and the size of each clustering
				change = 0
				Dist = [0 for x in range(k)]
				for j in range(n):
					#print "let's look at ", j, "th point"                
					# kDist is a list storing the distancce between given point and K centers
					KDist = [0 for y in range(k)]
					# dist is the distance between given point and ith center
					for i in range(k):
						#print "compare with", i, "th centroid"
						sub = array(Dataset[j])-array(Center[i])
						#print sub                    
						KDist[i] = sum([z**2 for z in sub])
					clu = KDist.index(min(KDist))
					#print clu                
					Dist[clu] += 1
					Sum1[clu] += array(Dataset[j])
					if(Label[j] != clu):
						change+= 1
						Label[j] = clu
				# update of centers
				for  i in range(k):
					Center[i] = Sum1[i]/float(Dist[i])
				print("new centers are :\n", Center)                
				print("new label are : ", Label)
		# if d equals 2, plot n points and k centers in form of diamond colorised according to their cluster
		if(d ==2):
			Color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
			Colors = [Color[i] for i in Label]
			for i in range(n):
				plt.scatter(Dataset[i][0],Dataset[i][1],color = Colors[i])
			for i in range(k):
				plt.scatter(Center[i][0],Center[i][1],color=Color[i],marker = 'D')
			plt.show()
		return(Center, Label)
	except ValueError:
		print("Not all data points are numeric")
		
		
## main function
a =[2,2]
b =[1,2]
c =[1,1]
d =[0,0]
f =[3,2]
Dataset = [a,b,c,d,f]
Dataset.append([1.5,0])
Dataset.append([3,4])
res = Kmeans(Dataset,2,2)
print(res)