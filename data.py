Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
 #准备数据：从文本中构建词向量
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],\
                 ['maybe','not','take','him','to','dog','park','stupid'],\
                ['my','dalmation','is','so','cute','r','love','him'],\
                ['stop','posting','stupid','worthless','garbage'],\
                ['mr','licks','ate','my','steak','how','to','stop','him'],\
                ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]   #由人工标注的文本类别，1代表侮辱性文字，0代表正常言论
    return postingList,classVec
    
#构建一个包含所有文档词汇且不重复的词汇表    
def createVocabList(dataSet):
    vocabSet=set([])               #创建一个空集
    for document in dataSet:            
        vocabSet=vocabSet|set(document)#创建两个集合的并集
    return list(vocabSet)  
    
#将每个文档转换成词向量
def setofWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)    #创建一个与词汇表等长且所有元素都为0的向量
    for word in inputSet:  #遍历文档
        if word in vocabList:       #若文档中出现词表中的词
            returnVec[vocabList.index(word)]=1 #则文档向量中的对应值设为1
        else:
            print ("the word: %s is not in my Vocabulary!" % word)
        return returnVec     #得到的与词汇表同等长度的词向量     
