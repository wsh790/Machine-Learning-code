import regression
import matplotlib.pyplot as plt
from numpy import *
xArr,yArr=regression.loadDataSet('ex0.txt')
# print (xArr[0:2])
# ws=regression.standRegres(xArr,yArr)
# # print (ws)
# xMat=mat(xArr)
yMat=mat(yArr)
# yHat=xMat * ws
# # fig =plt.figure()
# # ax=fig.add_subplot(111)
# # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# # xCopy=xMat.copy()
# # xCopy.sort(0)
# # yHat=xCopy*ws
# # ax.plot(xCopy[:,1],yHat)
# # plt.show()
# print (corrcoef(yHat.T,yMat))
yHat=regression.lwlrTest(xArr,xArr,yArr,0.003)
xMat=mat(xArr)
srtInd=xMat[:,1].argsort(0)
xSort=xMat[srtInd][:,0,:]
flg=plt.figure()
ax=flg.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten.A[0],s=2,c='red')
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')
plt.show()

