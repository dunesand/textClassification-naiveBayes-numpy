Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #加载数据
listOposts,listClasses=loadDataSet()
#构建词表
myVocabList=createVocabList(listOposts)
#构建文档矩阵
trainMat=[]
for postinDoc in listOposts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#测试用例
testEntry1=['love','my','dalmation']
testEntry2=['stupid','garbage']
thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
thisDoc2=array(setOfWords2Vec(myVocabList,testEntry2))
print(classifyNB(thisDoc1,trainMat,listClasses))
print(classifyNB(thisDoc2,trainMat,listClasses))
