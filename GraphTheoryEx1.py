
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import networkx as nx
import csv
import scipy
import scipy.sparse.csgraph as csg
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
from networkx.algorithms import tree


# In[15]:

#Question 2 - Floyd Warshall Algorithm
q1 = pd.read_csv("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/8020EX1Graph1.csv", header=None)


# In[16]:

adj =pd.DataFrame.as_matrix(q1,columns=None)


# In[17]:

#http://networkx.github.io/documentation/networkx-1.7/index.html
G1=nx.DiGraph(adj)
type(adj)


# In[18]:

adj2= np.asarray(adj, order='C') #Stack Overflow


# In[19]:

csg.floyd_warshall(adj2,directed = True)


# In[21]:

Floyd1=pd.DataFrame(csg.floyd_warshall(adj2,directed = True))
Floyd1.to_csv("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/Floyd1.csv", header=None)


# In[22]:

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html
csg.connected_components(adj, directed=True, connection='strong', return_labels=True)


# In[23]:

print (nx.number_strongly_connected_components(G1))


# In[24]:

[len(Gc) for Gc in sorted(nx.strongly_connected_component_subgraphs(G1),
                        key=len, reverse=True)]


# In[25]:

nx.draw(G1)
plt.show()


# In[26]:

sub=nx.strongly_connected_component_subgraphs(G1, copy=True)
type(sub)


# In[27]:


pos = nx.spring_layout(G1) 
  


# In[28]:

nx.draw_networkx_nodes(G1,pos,
                       nodelist=[2,69,3,75,12,76,13,77,14,78,22,79,23,85,24,86,32,87,33,88,34,89,35,95,36,96,37,97,38,98,39,99,43,44,45,46,47,48,49,54,55,56,57,58,59,65,66,67,68
],
                       node_color='r',
                       node_size=100,
                   alpha=0.5, label='Strong Component - 49 nodes')
nx.draw_networkx_nodes(G1,pos,
                       nodelist=[52,53,60,61,62,63,70,72,73,80,81,82,90,91,92],
                       node_color='b',
                       node_size=100,
                   alpha=0.5,label='Strong Component - 15 nodes')
nx.draw_networkx_nodes(G1,pos,
                       nodelist=[15,16,25,26],
                       node_color='g',
                       node_size=100,
                   alpha=0.5, label='Strong Component - 4 nodes')
nx.draw_networkx_nodes(G1,pos,
                       nodelist=[18,19,28,29],
                       node_color='m',
                       node_size=100,
                   alpha=0.5, label='Strong Component - 4 nodes')
nx.draw_networkx_nodes(G1,pos,
                       nodelist=[20,21,30,31],
                       node_color='c',
                       node_size=100,
                   alpha=0.5, label='Strong Component - 4 nodes')
nx.draw_networkx_nodes(G1,pos,
                       nodelist=[1,2,5,6,7,8,9,10,11,12,18,28,41,42,43,51,52,65,72,75,84,85,94,95],
                       node_color='w',
                       node_size=100,
                   alpha=0.5, label='Weakly connected/Cannot be reached')

nx.draw_networkx_edges(G1,pos,width=1.0,alpha=0.5)


# In[29]:

plt.axis('off')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,scatterpoints=1)
plt.savefig("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/StrongComponents.png",bbox_inches='tight') # save as png
plt.show()


# In[3]:

#Question 7b - Minimum spanning trees - Kruskal
q7 = pd.read_csv("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/8020EX1Graph2.csv", header=None)


# In[4]:

m7 =pd.DataFrame.as_matrix(q7,columns=None)
m7= np.asarray( m7, order='C')


# In[7]:

type(m7)
print(m7)


# In[6]:

#Python index starts at 0 for vertex '1'
g7=nx.from_numpy_matrix(m7)

g7.edges(data=True)


# In[48]:

mst_kruskal=nx.minimum_spanning_tree(g7,weight='weight')
print(sorted(mst_kruskal.edges(data=True)))


# In[63]:

labels=nx.draw_networkx_labels(mst_kruskal,pos=nx.spectral_layout(mst_kruskal))
nx.draw(mst_kruskal,pos=nx.spectral_layout(mst_kruskal))
plt.show()


# In[37]:

#another way to get kruskal, verify validity of existing package
g=Graph(15)
g.addEdge(0,1,7)
g.addEdge(1,2,3)
g.addEdge(3,4,3)
g.addEdge(4,5,5)
g.addEdge(6,7,3)
g.addEdge(7,8,7)
g.addEdge(9,10,1)
g.addEdge(10,11,6)
g.addEdge(12,13,4)
g.addEdge(13,14,5)
g.addEdge(2,5,2)
g.addEdge(5,8,3)
g.addEdge(8,11,5)
g.addEdge(11,14,2)
g.addEdge(1,4,2)
g.addEdge(4,7,7)
g.addEdge(7,10,3)
g.addEdge(10,13,7)
g.addEdge(0,3,3)
g.addEdge(3,6,2)
g.addEdge(6,9,1)
g.addEdge(9,12,4)


# In[38]:

# Python program for Kruskal's algorithm to find Minimum Spanning Tree
# of a given connected, undirected and weighted graph

from collections import defaultdict

#Class to represent a graph
class Graph:

	def __init__(self,vertices):
		self.V= vertices #No. of vertices
		self.graph = [] # default dictionary to store graph
		

	# function to add an edge to graph
	def addEdge(self,u,v,w):
		self.graph.append([u,v,w])

	# A utility function to find set of an element i
	# (uses path compression technique)
	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])

	# A function that does union of two sets of x and y
	# (uses union by rank)
	def union(self, parent, rank, x, y):
		xroot = self.find(parent, x)
		yroot = self.find(parent, y)

		# Attach smaller rank tree under root of high rank tree
		# (Union by Rank)
		if rank[xroot] < rank[yroot]:
			parent[xroot] = yroot
		elif rank[xroot] > rank[yroot]:
			parent[yroot] = xroot
		#If ranks are same, then make one as root and increment
		# its rank by one
		else :
			parent[yroot] = xroot
			rank[xroot] += 1

	# The main function to construct MST using Kruskal's algorithm
	def KruskalMST(self):

		result =[] #This will store the resultant MST

		i = 0 # An index variable, used for sorted edges
		e = 0 # An index variable, used for result[]

		#Step 1: Sort all the edges in non-decreasing order of their
		# weight. If we are not allowed to change the given graph, we
		# can create a copy of graph
		self.graph = sorted(self.graph,key=lambda item: item[2])
		#print self.graph

		parent = [] ; rank = []

		# Create V subsets with single elements
		for node in range(self.V):
			parent.append(node)
			rank.append(0)
	
		# Number of edges to be taken is equal to V-1
		while e < self.V -1 :

			# Step 2: Pick the smallest edge and increment the index
			# for next iteration
			u,v,w = self.graph[i]
			i = i + 1
			x = self.find(parent, u)
			y = self.find(parent ,v)

			# If including this edge does't cause cycle, include it
			# in result and increment the index of result for next edge
			if x != y:
				e = e + 1
				result.append([u,v,w])
				self.union(parent, rank, x, y)		 
			# Else discard the edge

		# print the contents of result[] to display the built MST
		print ("Following are the edges in the constructed MST")
		for u,v,weight in result:
			#print str(u) + " -- " + str(v) + " == " + str(weight)
			print ("%d -- %d == %d" % (u,v,weight))
			

#This code is contributed by Neelam Yadav
#http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/


# In[39]:

g.KruskalMST()


# In[77]:

#Question 7d - Dijkstra
length,path = nx.single_source_dijkstra(g7,0,weight='weight')
print (length)
type(path)


# In[97]:

#Question 8
#Weights adjusted to be -log(1-[prob of failure])
q8 = pd.read_csv("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/8020EX1Graph3.csv", header=None)
m8 =pd.DataFrame.as_matrix(q8,columns=None)
m8= np.asarray( m8, order='C')
type(m8)


# In[98]:

g8=nx.from_numpy_matrix(m8)
g8.edges(data=True)


# In[105]:

#Draw whole network
pos = nx.spring_layout(g8)
nx.draw(g8)
plt.show()


# In[99]:

mst_kruskal2=nx.minimum_spanning_tree(g8,weight='weight')
print(sorted(mst_kruskal2.edges(data=True)))
#Total MST weight = 0.110303792


# In[114]:

nx.draw_networkx_edges(g8,pos,
                       edgelist=[(0,13),(1,12),(2,6),(2,18),(3,14),(3,19),(4,10),(4,18),(5,11),(5,19),(6,15),(7,11),(7,15),(8,9),
                                 (8,13),(8,16),(12,16),(16,18),(17,19)],
                       edge_color='b',
                       alpha=0.5, label='Low failure')
nx.draw_networkx_edges(g8,pos,
                       edgelist=[(0,14),(0,15),(1,4),(1,5),(2,13),(3,7),(6,10),(9,17),(9,14),(10,11),(12,17)],
                       edge_color='r',
                       alpha=0.5,label='High failure')
nodes=nx.draw_networkx_nodes(g8,pos=nx.spring_layout(g8))
labels=nx.draw_networkx_labels(g8,pos=nx.spring_layout(g8))


# In[115]:

plt.axis('off')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,scatterpoints=1)
plt.savefig("C:/Users/Jess/OneDrive/Grad School/Graph Theory/EX1/Network.png",bbox_inches='tight') # save as png
plt.show()


# In[ ]:



