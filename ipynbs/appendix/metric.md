
# 度量

度量是集合中两个元素之间的距离函数,将集合的两个元素映射到非负实数域,$f(x_1, x_2) \rightarrow R$.要满足:
    1. 非负性,$f(x_1,x_2) >= 0$,取"=" 当且仅当 $x_1 = x_2$
    2. 对称性,$f(x_1,x_2) = f(x_2,x_1)$
    3. 三角不等式, $f(x_1,x_2) + f(x_2,x_3) >= f(x_1,x_3)$.  
我们在机器学习中会不可避免得遇到度量的选择问题.

## 点(向量)间的度量

最常见的度量发生在计算两个点之间的距离.也就是两个向量之间的距离.

### 对于数字型的数据

数字型的数据最常用的度量包括:

+ 闵可夫斯基距离(MinkowskiDistance)
    
    闵氏距离不是一种距离，而是一组距离的定义。

    $$ d = \sqrt[p] {\sum_{i=0}^n{|x_i-y_i|^p}} $$

    根据变参数p的不同，闵氏距离可以表示一类的距离。

    闵氏距离的特点:

    闵氏距离是由向量自身位置所决定的,它最大的好处是与样本无关,因此两个向量无论他们在哪个样本中,相互距离是一定的.这样就不会受到样本干扰

    闵氏距离的缺点:

     + 将各个分量的量纲(scale)，也就是“单位”当作相同的看待了。举个例子,一个人一次考试的成绩<语文分数,政治分数>,单位都是1分,但明显的这俩分数的价值是不同的.
     + 没有考虑各个分量的分布（期望，方差等)可能是不同的。


+ 欧式距离(Euclidean Distance)

我们一般不把向量作为线段了,而是用它的终点来表达它,这样两点离得越近自然就越是相似了,于是有了另一种常用的相似度量--**欧几里得距离**

![欧氏距离](../../source/img/欧式距离.png)

$$ d=\sqrt {\sum_{i=0}^n(x_i-y_i)^2}  $$

它是p=2时的闵可夫斯基距离

+ 欧几里得平方距离

    也就是欧几里得距离的平方,一般用于减小运算压力
    
    $$ d={\sum_{i=0}^n(x_i-y_i)^2}  $$
    

+  曼哈顿距离,我们可以想象田字形道路两点的距离

    同样是用于计算距离,像我们去某个地方一样,曼哈顿距离计算行进距离而不是直线距离,他的计算方式是这样:

    ![曼哈顿距离](../../source/img/曼哈顿距离.png)
    
    $$ d =  \sum_{i=1}^n |{x_i-y_i}| $$

    它是p=1时的闵可夫斯基距离
    
+ 切比雪夫距离(Chebyshev Distance)

    在扩展一步,曼哈顿距离只能横走竖走,我们加上斜走,就像象棋里面将帅的走法一样,这种距离算法就是切比雪夫距离算法:

    ![切比雪夫距离](../../source/img/切比雪夫距离.png)

    $$ d = max(|x_i-y_i|) $$

    它是p=inf时的闵可夫斯基距离
    

+ 最大距离,也就是无限范数诱导出的度量

    $$ \vert \vec x - \vec y \vert_\infty = \max \limits_{i =1}^d \vert x_i-y_i \vert $$
    

+ 马氏距离(Mahalanobis distance)

    考虑到向量分量之间的关系，数学上用协方差矩阵表示
    $$ d = \sqrt{ \{ \vec x - \vec y \}^T \times \Sigma^{-1}  \times \{ \vec x - \vec y \} } \vert $$
    其中 $ \Sigma $为数据点所在的向量空间的分量之间的协方差矩阵.

    与之相关的是***余弦相似度(Cosine Similarity)***
    
    我们可以只要关注向量间的夹角,也就是通过向量点积和向量模积的比值求出向量夹角的余弦
    $$ s = \frac {v1\dot v2}{\mod(v1) * \mod(v2)}$$
    
    **余弦相似度衡量的是维度间相对层面的差异**它往往关注的是维度是否被用到,而对用了多少并不敏感
    
    马氏距离度量的是长度，它只有远近之分.而余弦相似度是度量方向的,余弦相似度为1,你不可以说这两个向量一样,只能说他们是相似的,因为他们是同方向的.就好象(3,3)与(5,5).但是马氏距离度量的是长度,长度为0就可以认为他们是一样的.
    
## 概率分布之间的距离

P,Q是有共同事件空间的两个概率分布函数.

+ Helliger Distance
    离散情况下,  
    $H^2(P,Q) = \frac{1}{2} \sum_i (\sqrt p_i - \sqrt q_i)^2 $
    是一种对称的距离.
    
+ KLD(Kullback-Leibler Divergence) 
    离散情况下,  
    $D_{KL}(P||Q) = \sum_{i} p_i \log \frac{p_i}{q_i}$  
    从信息论的角度,KLD可以看成 真实分布为P时的P和Q的交叉熵 和 P的信息熵 之间的差.
    
+ JSD(Jensen-Shannon Divergence)
    离散情况下,  
    $JSD(P||Q) = \frac{1}{2} D_{KL}(P||Q) + \frac{1}{2}D_{KL}(Q||P)$
    JSD是一种基于KLD的,对称的距离,并且更一般地,JSD可以用于比较多个概率分布.
    $JSD_{\pi_1, \pi_2, ...,\pi_n} (P_1,P_2,...,P_n) = H(\sum_i \pi_i * P_i) - \sum_i \pi_i * H(P_i)$
    其中$H$为信息熵


### 对于文本或者非数字型的数据

对于文本或者非数字型的数据,一般我们会对这类数据先编码,将其编码为二进制数据,之后再计算编码间的距离.

最常见的编码方式是`one-hot`编码,具体就是将可枚举的非数字型数据编译为长度为枚举长度,其他全0而对应值为1的一串编码.当然编码方式千千万,很多时候我们也会自行编码.关于编码器解码器,这是另一个话题了.以后有机会再说.

最常用的二进制编码度量包括:

+ 汉明距离（Hamming distance）

    其定义为:两个等长向量对应位置的不同字符的个数,也可以说是将其中一个变为另外一个所需要作的最小替换次数。

    例如`(1111)`与`(1001)`之间的汉明距离为2。
    汉明距离常用于信息编码(为了增强容错性，应使得编码间的最小汉明距离尽可能大).

+ 莱温斯顿距离（Levenshtein distance）

    两个编码从一个变到另一个需要的最少的操作数.允许的操作包括添加,删除和替换,比如`(0111)`和`(1110)`距离为2,即`(1110)`头部添加`0`,尾部删除`0`
    
实际上以上两种度量都不限于处理二进制编码,他们同样可以用来比较字符串或者各种序列.


好的度量选择往往是提高模型的效果的必要条件.实践过程中倍加注意.

### ***使用scipy计算向量距离***

本文的代码中有这些距离的实现,但更加推荐的是使用scipy的代码计算,scipy是一系列用于科学计算的c代码的python封装.运行效率更高而且经过大量使用,代码更加健壮.

scipy中计算向量距离的接口是`scipy.spatial.distance.pdist(X, metric='euclidean', p=2, w=None, V=None, VI=None)`,其中

+ X是一组需要计算距离的向量
+ metric是度量,已经收录的度量包括:

度量|说明
---|---
braycurtis(u, v)|	计算两个一维数组的布雷-柯蒂斯距离
canberra(u, v)|计算两个一维数组的堪培拉距离
chebyshev(u, v)|	切比雪夫距离
cityblock(u, v)|	曼哈顿距离
correlation(u, v)|	计算两个一维向量的相关距离
cosine(u, v)|	计算两个一维向量Cosine距离
euclidean(u, v)|	计算两个一维向量欧式距离
hamming(u, v)|	计算两个一维向量的汉明距离
mahalanobis(u, v, VI)|	计算两个一维向量的马氏距离
minkowski(u, v, p)|计算两个一维向量的闵可夫斯基距离
seuclidean(u, v, V)|计算标准化欧氏距离
jaccard(u, v)|	计算两个一维布尔向量的jaccard距离
sqeuclidean(u, v)|计算欧氏平方距离
wminkowski(u, v, p, w)|	计算加权闵可夫斯基距离
yule(u, v)|计算两个一维布尔向量的Yule dissimilarity
sokalmichener(u, v)|计算两个一维布尔向量的Sokal-Michener dissimilarity
sokalsneath(u, v)|计算两个一维布尔向量的Sokal-Sneath dissimilarity
rogerstanimoto(u, v)|计算两个一维布尔向量的Rogers-Tanimoto dissimilarity
russellrao(u, v)|计算两个一维布尔向量的Russell-Rao dissimilarity 
matching(u, v)|	计算两个一维布尔向量的Matching dissimilarity
kulsinski(u, v)|计算两个布尔型一维数组之间的kulsinski系数
dice(u, v)|	计算两个布尔型一维数组之间的Dice系数

+ 其他的参数则是不同metric的超参

### 演示 


```python
from scipy.spatial.distance import pdist
```


```python
X = [[1,2],[3,4]]
```


```python
pdist(X, metric='euclidean')
```




    array([ 2.82842712])



## 集合间的度量


基于点和点之间的度量$d$,我们就可以规定集合A,B之间的距离.常用的有以下几种：

+ 最大距离（Complete-linkage clustering）

    $$ \max\{ d(a,b) : a \in A, b\in B \} $$

+ 最小距离(Single-linkage clustering)

    $$ \min\{ d(a,b) : a \in A, b\in B \}$$

+ 平均距离(UPGMA)

    $$ \frac {1} {n\times m} \sum \limits_ {a \in A, b\in B} d(a,b) \} $$
    其中$ n,m $分别是A,B聚类的基数，也就是包含的点的数量.

+ 中点距离(UPGMC),$$ \vert d(C_A,C_B) \vert $$,其中$ C_A, C_B $分别是A，B的中点，也就是离中心最近的点

+ 能量距离（Energy distance）,$$ \frac {2} {n\times m} \sum \limits_ {a \in A, b\in B} \vert a-b\vert_2 - \frac{1}{n^2} \sum \limits_{a_i,a_j \in A} \vert a_i -a_j \vert_2 -\frac{1}{m^2} \sum \limits_{b_i,b_j \in B} \vert b_i -_j \vert_2 \}$$

