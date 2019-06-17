from numpy import *
import logRegres

dataArr,labelMat=logRegres.loadDataSet()
# print(logRegres.gradAscent(dataArr,labelMat))

# weights = logRegres.gradAscent(dataArr,labelMat)
weights=logRegres.sGd(array(dataArr),labelMat)
logRegres.plotBestFit(weights)