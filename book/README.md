# 引言

建立这个项目，是为了梳理和总结传统机器学习（Machine Learning）方法（methods）或者算法(algo),和各位同仁相互学习交流。

机器学习是人工智能（Artificial Intelligence）的一个分支，，也是实现人工智能最重要的手段。区别于传统的基于规则（rule-based）的算法，机器学习可以从数据中获取知识，从而实现规定的任务[Ian Goodfellow and Yoshua Bengio and Aaron Courville的《Deep Learning》]。这些知识可以分为四种：1.总结（summarization）；2.预测(prediction)；3.估计(estimation)；4.假想验证(hypothesis testing)。机器学习主要关心的是预测[Varian在《Big Data : New Tricks for Econometrics》]，预测的可以是连续性的输出变量、分类、聚类或者物品之间的有趣关联。

根据数据配置（setting，是否有标签，可以是连续的也可以是离散的）和任务，我们可以将机器学习方法分为三种有监督（supervised）、无监督（unsupervised）、半监督（semi-supervised）和强化学习（reinforcement learning）。有监督学习指的是数据集有给定的输出,也就是加了标签，根据输出类型可以分为回归（regression，当输出为连续型）和分类（classification，输出为离散型）。无监督学习是指没有给定的输出，主要任务是做聚类分析，关心的是模式（patterns）和数据点之间的区别度（dissimilarity）。半监督学习介于监督学习和半监督学习之间，从任务和技术角度属于监督学习。但很多场景下，给数据加标签的成本比较大，所以出现了半监督学习，来使用大量的没有加标签的数据。强化学习则是通过反馈来采取有利的下一步行动，目标和手段上都区别于传统的监督、非监督学习。

我的项目按照聚类、分类回归和关联法则，分为三个章节。其中聚类属于非监督学习，分类回归属于监督学习。而关联法则在诞生时属于非监督学习，用来发现物品之间有趣的关联，也就是模式（pattern），后来我们也发展出了它作为分类器的应用，属于监督学习的场景。

具体涉及的方法算法中，聚类有K-means方法（主要是Lloyd算法），层次聚类方法（Hierarchical Clustering）和高斯混合模型（GMM，Gaussian Mixture Model）；分类回归包括朴素贝叶斯分类器（Naive Bayes Classifier）、决策树（Decision Tree）、随机森林（Random Forest）、支持向量机（SVM，Support Vector Machine）、逻辑回归（Logistics Regression）和KNN(K Nearest Neighbors）；关联规则则有A Priori和FP Growth（Frequent Pattern Growth）。

每个方法/算法，我主要介绍方法、背后的逻辑、复杂度、收敛性分析、相关算法的步骤，会穿插一些机器学习通用技巧以及与相似算法的比较。

本人理论和经验尚浅，在写作过程中也是一边总结一边学习，复杂度和收敛性分析更是第一次尝试着做，难免会有很多生涩稚嫩甚至错误的地方。恳请各位前辈和有志同仁谅解和赐教。如有模糊不明或者值得讨论之处，也欢迎大家留言评论。