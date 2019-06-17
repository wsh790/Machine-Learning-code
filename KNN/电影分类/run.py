import KNN
from numpy import *
import operator
group,labels=KNN.createDataSet()

def classify0(input,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(input,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range (k):
        voteLabel=labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),
                            reverse=True
                            )
    return sortedClassCount[0][0]

if __name__ == '__main__':


    test = [101,20]


    test_class = classify0(test, group, labels, 3)


    print(test_class)