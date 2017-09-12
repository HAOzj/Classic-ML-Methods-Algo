# 引言

建立这个项目,是为了梳理和总结传统机器学习(Machine Learning)方法(methods)或者算法(algo),和各位同仁相互学习交流.

机器学习是人工智能(Artificial Intelligence)的一个分支,也是实现人工智能最重要的手段.区别于传统的基于规则(rule-based)的算法,机器学习可以从数据中获取知识,从而实现规定的任务[Ian Goodfellow and Yoshua Bengio and Aaron Courville的***Deep Learning***].这些知识可以分为四种:

1. 总结（summarization）
2. 预测(prediction)
3. 估计(estimation)
4. 假想验证(hypothesis testing)

机器学习主要关心的是预测[Varian在***Big Data : New Tricks for Econometrics***],预测的可以是连续性的输出变量,分类,聚类或者物品之间的有趣关联.

根据数据配置(setting,是否有标签，可以是连续的也可以是离散的)和任务目标,我们可以将机器学习方法分为四种:

+ 无监督(unsupervised)

    训练数据没有给定的输出,主要任务是做聚类分析,关心的是模式(patterns)和数据点之间的区别度(dissimilarity).

+ 有监督(supervised)

    训练数据集有给定的输出,也就是加了标签,根据输出类型可以分为回归(regression，当输出为连续型)和分类(classification,输出为离散型)
    

+ 半监督(semi-supervised)

    半监督学习介于监督学习和半监督学习之间,从任务和技术角度属于监督学习.但很多场景下,给数据加标签的成本比较大,所以出现了半监督学习,来使用大量的没有加标签的数据.
    
+ 强化学习(reinforcement learning)

    强化学习则是通过反馈来采取有利的下一步行动,目标和手段上都区别于传统的监督,非监督学习.



该项目按照以上分类方式,分为四个章节.每一章节中会有典型方法和典型应用场景.涉及到的方法有:

+ 无监督

    + K-means方法(主要是Lloyd算法)
    + 层次聚类方法(Hierarchical Clustering)
    + 高斯混合模型(GMM，Gaussian Mixture Model)
    + A Priori关联规则
    + FP Growth(Frequent Pattern Growth)关联规则
    
+ 有监督

    + KNN(K Nearest Neighbors）
    + 朴素贝叶斯分类器(Naive Bayes Classifier)
    + 决策树(Decision Tree)
    + 随机森林(Random Forest)
    + 最小二乘法回归
    + 逻辑回归(Logistics Regression)
    + 支持向量机(SVM,Support Vector Machine)
    + 协同过滤(Collaborative Filtering)
    
    

每个方法/算法,我主要介绍方法,背后的逻辑,相关算法的步骤,再进行复杂度和收敛性分析,会穿插一些机器学习通用技巧以及与相似算法的比较.

本人理论和经验尚浅,在写作过程中也是一边总结一边学习,复杂度和收敛性分析更是第一次尝试着做,难免会有很多生涩稚嫩甚至错误的地方.恳请各位前辈和有志同仁谅解和赐教.如有模糊不明或者值得讨论之处,也欢迎大家留言评论.
