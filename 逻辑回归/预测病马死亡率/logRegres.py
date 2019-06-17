from numpy import *

def loadDataSet():
    dataMat=[];labelMat=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weigths=ones((n,1))
    for k in range (maxCycles):
        h=sigmoid(dataMatrix*weigths)
        error=(labelMat-h)
        weigths=weigths+alpha*dataMatrix.transpose()*error
    return weigths

def sGd(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range (m):
            alpha=4/(i+j+1.0)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=float(classLabels[randIndex])-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    m=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range (m):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()

def classify(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in  frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(currLine[-1])
    trainWeights=sGd(array(trainingSet),array(trainingLabels),500)
    errorCount=0; numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range (len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classify(array(lineArr),array(trainWeights)))!=int(currLine[-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)*100
    print ("the error rate of the test is : %f "%errorRate)
    return errorRate

def multiTest():
    numTest=10
    errorSum=0.0
    for k in range(numTest):
        errorSum+=colicTest()
    print("after %d iterations, the average error ratr is : %f"%(numTest,float(errorSum)/numTest))








