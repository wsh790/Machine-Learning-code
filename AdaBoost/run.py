import adaboost
from numpy import *
# dataMat,classLabels=adaboost.loadSimpData()
# print dataMat,classLabels
# D=mat(ones((5,1))/5)
# print (adaboost.buildStump(dataMat,classLabels,D))
dataArr,classLabels=adaboost.loadDataSet('horseColicTraining2.txt')
classifierArray,aggClassEst=adaboost.adaBoostTrainDS(dataArr,classLabels,10)

# print adaboost.adaClassify([0,0],classifierArray)
# testArr,testLabelsArr=adaboost.loadDataSet('horseColicTest2.txt')
# prediction=adaboost.adaClassify(testArr,classifierArray)
# errArr=mat(ones((67,1)))
# print(errArr[prediction!=mat(testLabelsArr).T].sum())
adaboost.plotROC(aggClassEst.T,classLabels)