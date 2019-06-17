import kMeans
from numpy import *

dataMat=mat(kMeans.loadDataSet('testSet.txt'))
# print min(dataMat[:,0])
#
# print(kMeans.randCent(dataMat,2))
#
# print(kMeans.distEclud(dataMat[0],dataMat[1]))

myCentroids,clustAssing = kMeans.kMeans(dataMat,4)
print myCentroids