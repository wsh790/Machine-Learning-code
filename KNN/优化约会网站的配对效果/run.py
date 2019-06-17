import KNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
#
# datingDataMat,datingLabels=KNN.file2matrix('datingTestSet2.txt')
# normMat,ranges,minVals=KNN.autoNorm(datingDataMat)

# print (normMat)
# print(ranges)
# print(minVals)

# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()
KNN.classifyPerson()