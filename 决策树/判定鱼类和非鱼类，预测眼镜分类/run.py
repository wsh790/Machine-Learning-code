import trees
import treePlotter
# myDat,labels=trees.createDataSet()
# myDat[0][-1]='maybe'
# print (myDat)
# shannonEnt=trees.calcShannonEnt(myDat)
# print(shannonEnt)
# print(trees.splitDataSet(myDat,0,1))
# print(trees.chooseBestFeatureToSplit(myDat))
# print(myDat)
# myTree=trees.createTree(myDat,labels)
# print (myTree)
# treePlotter.createPlot()

# print(treePlotter.retrieveTree(1))
# myTree=treePlotter.retrieveTree(0)
# print(treePlotter.getNumLeafs(myTree))
# print(treePlotter.getTreeDepth(myTree))
# treePlotter.createPlot(myTree)
# print(trees.classify(myTree,labels,[1,0]))
# print(trees.classify(myTree,labels,[1,1]))

# trees.storeTree(myTree,'classifierStorage.txt')
# print(trees.grabTree('classifierStorage.txt'))

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tarRate']
lensesTree=trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)
