Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #trainMatrix已转换为向量的文档矩阵，trainCategory每篇文档类别标签所构成的向量
def trainNB(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)  #文档数量
    numWords=len(trainMatrix[0])  #第一篇文档的长度，也就是词汇表的长度
    pAbusive=sum(trainCategory)/float(numTrainDocs) #侮辱文档占总文档比例
    #初始化概率
    p0Num=zeros(numWords)         #初始化概率
    p1Num=zeros(numWords)       
    p0Denom=0
    p1Denom=0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:          #如果是侮辱性文档
            p1Num+=trainMatrix[i]        #文档对应的词语数量全部加1，向量相加
            p1Denom+=sum(trainMatrix[i]) #侮辱性文档词语的总数量
        else:
            p0Num+=trainMatrix[i]        #正常文档对应的词语数量向量
            p0Denom+=sum(trainMatrix[i]) #正常文档词语的总数量
    #对p1Num的每个元素做除法，即侮辱性文档中出现每个词语的概率
    p1Vect=p1Num/p1Denom    #change to log()
    #对p0Num的每个元素做除法，即正常文档中出现每个词语的概率
    p0Vect=p0Num/p0Denom    #change to log()    
    return p0Vect,p1Vect,pAbusive

>>> #朴素贝叶斯分类函数
#vec2Classify待分类文档向量
#trainMatrix训练文档矩阵
#trainCategory训练文档类别向量
def classifyNB(vec2Classify,trainMatrix,trainCategory):
    p0Vect,p1Vect,pAb=trainNB(trainMatrix,trainCategory)
    #计算待分类文档词条对应的条件概率P(w0|Ci)P(w1|Ci)P(w2|Ci)...P(wn|Ci)
    p1VectClassify=vec2Classify*p1Vect   #元素相乘
    p0VectClassify=vec2Classify*p0Vect
    #去掉列表中的0元素
    p1Cond=[];p0Cond=[]
    for i in range(len(p1VectClassify)):
        if p1VectClassify[i]==0:
            continue
        else:
            p1Cond.append(p1VectClassify[i])
    for i in range(len(p0VectClassify)):
        if p0VectClassify[i]==0:
            continue
        else:
            p0Cond.append(p0VectClassify[i])
    #计算各概率向量的乘积
    if len(p0Cond):                        #若p0Cond不为空，即p0VectClassify不全为0
        pC0=reduce(operator.mul, p0Cond, 1)#计算概率向量内元素乘积，得到P(w|C0)
    else:                      
        pC0=0                   
    if len(p1Cond):
        pC1=reduce(operator.mul, p1Cond, 1)
    else:
        pC1=0
    p1=pC1*pAb
    p0=pC0*(1.0-pAb)
    if p1 > p0:
        return 1 
    else:
        return 0
