from numpy import *
import matplotlib.pyplot as plt



def loadSimpData():
    datMat= matrix([[1.,2.1], [2.,1.1], [1.3,1.], [1.,1.], [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def loadDataSet(filename):
    numFeat=len(open(filename).readline().strip().split('\t'))
    dataMat = []
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        linArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            linArr.append(float(curLine[i]))
        dataMat.append(linArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq =='lt':
        retArray[dataMatrix[:,dimen]<=threshVal] =-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStrump={}
    bestClassEst=mat(zeros((m,1)))
    minError=inf
    for i in range (n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin +float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T * errArr
                #print "split: dim %d , thresh %.2f, thresh inequal: %s, the weighted error is %.3f"%(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst=predictedVals.copy()
                    bestStrump['dim']=i
                    bestStrump['thresh']=threshVal
                    bestStrump['ineq']=inequal
    return bestStrump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabel,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabel,D)
        #print 'D:', D.T
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #print 'classEst', classEst.T
        expon=multiply(-1*alpha*mat(classLabel).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst += alpha * classEst
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabel).T,ones((m,1)))
        #aggErrors = sign(aggClassEst) != mat(classLabel).T
        errorRate = aggErrors.sum()/m
        print ('total error: ', errorRate,'\n')
        if errorRate==0.0 :
            break
    return weakClassArr,aggClassEst

def adaClassify(dataToClass,classifierArr):
    dataMatrix=mat(dataToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range (len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha'] * classEst
        #print aggClassEst
    return sign(aggClassEst)

def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()
    flg=plt.figure()
    flg.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False postive rate')
    plt.ylabel('True positive rate')
    plt.title ('ROC Curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the area under curve is: " ,ySum*xStep)




