from numpy import *
import regTrees
# testMat=mat(eye(4))
# print testMat
# mat0,mat1=regTrees.binSplitDataSet(testMat,1,0.5)
# print mat1
myDat=regTrees.loadDataSet('ex00.txt')
print (regTrees.createTree(myDat))