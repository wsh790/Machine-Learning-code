from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet :
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print "the word : %s is not in my Vocabulary" %word
    return returnVec

# trainMatrix---setOfWords2Vec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range (numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix [i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix [i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num /p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):

    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)

    if p1 > p0:
        return 1
    else:
        return 0

#convenience function

def testingNB():
    listPosts,listClasses = loadDataSet()
    myVocabList=createVocabList(listPosts)
    trainMat=[]
    for post in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,post))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry1=['love','my','dalmation']
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print testEntry1, 'classified as:' ,classifyNB(thisDoc1, p0V, p1V, pAb)

    testEntry2=['stupid','garbage']
    thisDoc2=array(setOfWords2Vec(myVocabList,testEntry2))
    print testEntry2,'classified as:',classifyNB(thisDoc2,p0V,p1V,pAb)



