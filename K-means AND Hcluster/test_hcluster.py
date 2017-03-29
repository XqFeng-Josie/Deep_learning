#coding=utf-8
#层次聚类测试
#存在问题  2017/03/29 16:03
from Hierarchical_clustering import *
subjects=[[0.0, 0.8, 0.0, 0.4, 0.7, 0.7, 0.1, 0.0],
          [0.5, 0.8, 1.0, 0.0, 0.5, 0.9, 0.7, 0.2],
          [0.1, 0.2, 0.5, 0.4, 0.6, 0.7, 0.7, 0.8],
          [0.6, 0.5, 1.0, 0.9, 0.9, 0.4, 0.2, 0.5],
          [0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.9, 0.5],
          [0.0, 0.4, 1.0, 0.9, 0.9, 0.3, 0.3, 0.8],
          [0.4, 0.8, 0.2, 0.2, 0.0, 0.8, 0.0, 0.3],
          [0.2, 0.3, 0.9, 0.3, 0.3, 0.9, 0.1, 0.1],
          [0.4, 0.6, 0.4, 0.0, 0.7, 0.3, 0.0, 0.0],
          [0.7, 1.0, 0.6, 0.9, 0.1, 0.0, 1.0, 0.2]]
for i in range(len(subjects)):
   cluster_node(subjects[i],left=None, right=None, distance=0.0, id=i, count=1)

clust = hcluster(subjects)
printclust(clust, labels=None, n=0)
print "Deep searching list:"
print  get_cluster_elements(clust)
print "Depth is =",str(getdepth(clust))
print getheight(clust)
