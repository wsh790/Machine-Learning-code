from numpy import *
from numpy import linalg as la

def euclidSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

def parasSim(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num=float(inA.T,inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0:
            continue
        overLap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap)==0:
            similarity=0
        else:
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])
        simTotal +=similarity
        ratSimTotal+=similarity*userRating
    if simTotal ==0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unRatedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unRatedItems)==0:
        return 'you rated everything'
    itemScores=[]
    for item in unRatedItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,k=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    U,sigma,VT=la.svd(dataMat)
    Sig4=mat(eye(4)*sigma[:4])
    xforcedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item :
            continue
        similarity=simMeas(xforcedItems[item,:].T,xforcedItems[j,:].T)
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal





