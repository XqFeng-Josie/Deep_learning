#coding=utf-8
#层次聚类Hierarchical_clustering方法实现
from numpy import *
from scipy.cluster._hierarchy import cluster_dist
from math import *


class cluster_node:
    # 构造函数:self:this
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.left = left
        self.right = right# 每次聚类都是一对数据，left保存其中一个数据，right保存另一个
        self.vec = vec # 保存两个数据聚类后形成新的中心
        self.id = id
        self.distance = distance
        self.count = count

#返回两个结点之间的距离
def L2dist(v1, v2):
    return sqrt(sum((array(v1) - array(v2)) ** 2))


def L1dist(v1, v2):
    return sum(abs(v1 - v2))


# def Chi2dist(v1,v2):
#    return sqrt(sum((v1-v2)**2)

def hcluster(rows, distance=L2dist):
    distances = {}
    currentclustid = -1

    # Clusters are initially just the rows
   #初始化聚类矩阵
    clust = [cluster_node(rows[i], id=i) for i in range(len(rows))]
    while len(clust) > 1:
        # 初始化最近结点
        lowestpair = (0, 1)
        # 初始化最近结点之间的距离
        closest = distance(clust[0].vec, clust[1].vec)
        # loop through every pair looking for the smallest distance
        # 计算两两结点之间的距离
        for i in range(len(clust)-1):
            for j in range(i + 1, len(clust)):
                # distances is the cache of distance calculations
                # 计算过的点加入distances字典里
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                # 更新最短距离
                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        # calculate the average of the two clusters
        bic1, bic2 = lowestpair  # 解包bic1 = i , bic2 = j
        # 形成新的类中心
        mergevec = [(clust[bic1].vec[i] + clust[bic2].vec[i]) / 2.0 for i in range(len(clust[0].vec))]
        # create the new cluster 二合一
        newcluster = cluster_node(array(mergevec), left=clust[bic1],
                                  right=clust[bic2],
                                  distance=closest, id=currentclustid)
        # cluster ids that weren't in the original set are negative
        currentclustid -= 1
        # 删除聚成一起的两个数据，由于这两个数据要聚成一起
        del clust[bic2]
        del clust[bic1]
        #
        # 补回新聚类中心
        clust.append(newcluster)
    return clust[0]


def extract_clusers(clust_dist):
    return cl + cr

#深度优先搜素
def get_cluster_elements(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)
        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        print cl, cr
        return cl + cr
def get_cluster_elements2(clust):
    if clust.id >= 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left != None:
            cl = get_cluster_elements(clust.left)

        if clust.right != None:
            cr = get_cluster_elements(clust.right)
        print cl,cr
        return cl + cr

def printclust(clust, labels=None, n=0):
    # indent to make a hierarchy layout
    for i in range(n): print ' ',
    if clust.id < 0:
        # negative id means that this is branch
        print '-'
    else:
        # positive id means that this is an endpoint
        if labels == None:
            print clust.id
        else:
            print labels[clust.id]
    # now print the right and left branches
    if clust.left != None: printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None: printclust(clust.right, labels=labels, n=n + 1)


def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left == None and clust.right == None: return 1
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left) + getheight(clust.right)


def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left == None and clust.right == None: return 0

    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance