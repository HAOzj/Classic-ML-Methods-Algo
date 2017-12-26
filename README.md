# Classic-ML-Methods-Algo

# 引言

建立这个项目,是为了梳理和总结传统机器学习(Machine Learning)方法(methods)或者算法(algo),和各位同仁相互学习交流.

现在的深度学习本质上来自于传统的神经网络模型,很大程度上是传统机器学习的延续,同时也在不少时候需要结合传统方法来实现.

任何机器学习方法基本的流程结构都是通用的;使用的评价方法也基本通用;使用的一些数学知识也是通用的.

本文在梳理传统机器学习方法算法的同时也会顺便补充这些流程,数学上的知识以供参考.

## 机器学习

机器学习是人工智能(Artificial Intelligence)的一个分支,也是实现人工智能最重要的手段.区别于传统的基于规则(rule-based)的算法,机器学习可以从数据中获取知识,从而实现规定的任务[Ian Goodfellow and Yoshua Bengio and Aaron Courville的***Deep Learning***].这些知识可以分为四种:

1. 总结（summarization）
2. 预测(prediction)
3. 估计(estimation)
4. 假想验证(hypothesis testing)

机器学习主要关心的是预测[Varian在***Big Data : New Tricks for Econometrics***],预测的可以是连续性的输出变量,分类,聚类或者物品之间的有趣关联.

## 机器学习分类

根据数据配置(setting,是否有标签，可以是连续的也可以是离散的)和任务目标,我们可以将机器学习方法分为四种:

+ 无监督(unsupervised)

    训练数据没有给定的输出,主要任务是做聚类分析,关心的是模式(patterns)和数据点之间的区别度(dissimilarity).

+ 有监督(supervised)

    训练数据集有给定的输出,也就是加了标签,根据输出类型可以分为回归(regression，当输出为连续型)和分类(classification,输出为离散型)
    

+ 半监督(semi-supervised)

    半监督学习介于监督学习和半监督学习之间,从任务和技术角度属于监督学习.但很多场景下,给数据加标签的成本比较大,所以出现了半监督学习,来使用大量的没有加标签的数据.
    
+ 强化学习(reinforcement learning)

    强化学习则是通过反馈来采取有利的下一步行动,目标和手段上都区别于传统的监督,非监督学习.

## 本项目编排

该项目按照以上分类方式,分为三个章节.每一章节中会有典型方法和典型应用场景,其中关联法则在诞生时属于非监督学习，用来发现物品之间有趣的关联,也就是模式（pattern）,后来我们也发展出了它作为分类器的应用,属于监督学习的场景.涉及到的方法有:

+ 无监督
    + K-means方法(主要是Lloyd算法)
    + 层次聚类方法(Hierarchical Clustering)
    + 高斯混合模型(GMM，Gaussian Mixture Model)
    
+ 有监督
    + 朴素贝叶斯分类器(Naive Bayes Classifier)
    + 感知机(Perceptron)
    + 支持向量机(SVM,Support Vector Machine)
    + 逻辑回归(Logistics Regression)
    + KNN(K Nearest Neighbors）
    + 决策树(Decision Tree)
    + 随机森林(Random Forest)

+ 关联规则
    + A Priori关联规则
    + FP Growth(Frequent Pattern Growth)关联规则

所有方法/算法的介绍都按照以上三个分类,放在ipynb文件夹下相应的文件夹里,代码实现放在code文件夹里.

每个方法/算法,主要介绍方法,背后的逻辑,相关算法的步骤,再进行复杂度和收敛性分析,会穿插一些机器学习通用技巧以及与相似算法的比较.


同时在附录中会包含内容:

+ 数据抽样

+ 数据预处理
    + 归一化
    + 白化
    + 数据增强
    + 连续数据离散化
    + 离散数据向量化
    + 数据平滑化
    + 特征提取
    + 特征组合
    + 特征降维

+ 模型评价
    + 分类模型评估
    + 回归模型评估
    + 聚类模型评估

+ 超参选择
    + grid search
    + 基于贪婪算法的 grid search 改进
    + 基于动态规划的 grid search 改进

+ 模型组合
    + bagging
    + boosting 

+ 背景知识
    + 决策论基础
    + 信息论基础
    + 图论基础
    + 概率分布
    + 贝叶斯方法
    + 距离度量
    + 线性优化
    + 非线性优化
    + 纬度灾难
    + 过拟合

全文并不基于某种语言或者工具,而是希望讲解其中的原理.但要描述一个算法最好的方式其实是将其用编程语言实现一遍.
本文使用python来实现这些算法,毕竟python的语法相对简洁可读性强.

另外本文也会将对应的python标准数据科学工具箱(nmupy,scipy,pandas,etc)中的接口写出来方便查询.

## 文件位置

各个模型/算法的介绍和实现都在ipynbs文件夹下.

## 致谢

这个项目的开始受到了@黄思喆的启发,感谢@黄思哲 帮忙排版，@金岩@钟志强@李卓 等人的支持和鼓励,也感谢碰巧关注这个项目的同道中人.本人理论和经验尚浅,在写作过程中也是一边总结一边学习,复杂度和收敛性分析更是第一次尝试着做,难免会有很多生涩稚嫩甚至错误的地方.恳请各位前辈和有志同仁谅解和赐教.如有模糊不明或者值得讨论之处,也欢迎大家留言评论.


## 用法

这里我们逐步解释如何将内容编译成静态网页.

### 环境初始化

+ 没有`node.js`环境的先安装`node.js`环境
+ 第一次pull下来后,先cd到`book目录`,使用命令`npm install`安装环境
+ 在`book目录`下,使用`gitbook install`安装gitbook插件

### 流程

1. 在ipynbs文件夹下写好内容
2. `file->download as->markdown(.md)`将文章导出为markdown并下载
3. 将导出的文件放入`book目录`下的合适位置(最好按章节放)
4. 编辑`book目录`下的`SUMMARY.md`文件,按格式规划好文章目录并将文件注册到其中,具体看`SUMMARY.md`文件内容依葫芦画瓢
5. `book目录`下运行`gitbook build`
6. 将`book/_book`目录下的内容复制到`docs`文件夹下
7. 使用`git`上传变更
