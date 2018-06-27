
#### Spark机器学习库简介
MLlib是Spark的机器学习库。旨在简化机器学习的工程实践工作，并方便扩展到更大规模。MLlib由一些通用的学习算法和工具组成，包括分类、回归、聚类、协同过滤、降维等，同时还包括底层的优化原语和高层的管道API。

它提供如下工具：

1. 机器学习算法：常规机器学习算法包括分类、回归、聚类和协同过滤。
2. 特征工程：特征提取、特征转换、特征选择以及降维。
3. 管道：构造、评估和调整的管道的工具。
4. 存储：保存和加载算法、模型及管道。
5. 实用工具：线性代数，统计，数据处理等。

#### Spark机器学习库MLlib和ML
Spark机器学习库从 1.2 版本以后被分为两个包，分别是：

+ spark.mllib 包含基于RDD的原始算法API。
Spark MLlib 历史比较长，1.0 以前的版本中已经包含了，提供的算法实现都是基于原始的 RDD，从学习角度上来讲，其实比较容易上手。如果您已经有机器学习方面的经验，那么您只需要熟悉下 MLlib 的 API 就可以开始数据分析工作了。想要基于这个包提供的工具构建完整并且复杂的机器学习流水线是比较困难的。

+ spark.ml 则提供了基于DataFrames 高层次的API，可以用来构建机器学习管道。
Spark ML Pipeline 从 Spark1.2 版本开始，成为可用并且较为稳定的新的机器学习库。ML Pipeline 弥补了原始 MLlib 库的不足，向用户提供了一个基于 DataFrame 的机器学习工作流式 API 套件，使用 ML Pipeline API，我们可以很方便的把数据处理，特征转换，正则化，以及多个机器学习算法联合起来，构建一个单一完整的机器学习流水线。显然，这种新的方式给我们提供了更灵活的方法，而且这也更符合机器学习过程的特点。

从官方文档来看，Spark ML Pipeline 虽然是被推荐的机器学习方式，但是并不会在短期内替代原始的 MLlib 库，因为 MLlib 已经包含了丰富稳定的算法实现，并且部分 ML Pipeline 实现基于 MLlib。


在Spark2.0中，spark.mllib包中的RDD接口已进入维护模式。现在主要的机器学习接口为spark.ml包中的基于数据框接口。

这一转变包含哪些信息？

1. MLlib将继续在spark.mllib中支持基于RDD的接口。
2. MLlib不会向基于RDD的接口中继续添加新的特征。
3. 在Spark2.0以后的版本中，将继续向基于数据框的接口添加新特征以缩小与基于RDD接口的差异。
4. 当两种接口之间达到特征相同时（初步估计为Spark2.2），基于RDD的接口将被废弃。
5. 基于RDD的接口将在Spark3.0中被移除。

为什么MLlib转向DataFrames接口？

1. 数据框可以提供比RDD更容易掌握使用的接口。数据框的主要优点包括Spark数据源来源、结构化查询语言的数据框查询、各编程语言之间统一的接口。
2. 基于数据框的MLlib接口为多种机器学习算法与编程语言提供统一的接口。
3. 数据框有助于实现机器学习管道，特别是特征转换。

#### Spark机器学习库的功能
下面的列表列出了两个包的主要功能。

##### [spark.mllib: 数据类型，算法以及工具](http://spark.apache.org/docs/latest/mllib-guide.html)
Data types（数据类型）
Basic statistics（基础统计）
summary statistics（摘要统计）
correlations（相关性）
stratified sampling（分层抽样）
hypothesis testing（假设检验）
streaming significance testing
random data generation（随机数据生成）
Classification and regression（分类和回归）
linear models (SVMs, logistic regression, linear regression)（线性模型（SVM，逻辑回归，线性回归））
naive Bayes（朴素贝叶斯）
decision trees（决策树）
ensembles of trees (Random Forests and Gradient-Boosted Trees)（树套装（随机森林和梯度提升决策树））
isotonic regression（保序回归）
Collaborative filtering（协同过滤）
alternating least squares (ALS)（交替最小二乘（ALS））
Clustering（聚类）
k-means（K-均值）
Gaussian mixture（高斯混合）
power iteration clustering (PIC)（幂迭代聚类（PIC））
latent Dirichlet allocation (LDA)（隐含狄利克雷分配）
bisecting k-means（平分K-均值）
streaming k-means（流式K-均值）
Dimensionality reduction（降维）
singular value decomposition (SVD)（奇异值分解（SVD））
principal component analysis (PCA)（主成分分析（PCA））
Feature extraction and transformation（特征抽取和转换）
Frequent pattern mining（频繁模式挖掘）
FP-growth（FP-增长）
association rules（关联规则）
PrefixSpan（PrefixSpan）
Evaluation metrics（评价指标）
PMML model export（PMML模型导出）
Optimization (developer)（优化（开发者））
stochastic gradient descent（随机梯度下降）
limited-memory BFGS (L-BFGS)（有限的记忆BFGS（L-BFGS））

##### [spark.ml: 机器学习管道高级API](http://spark.apache.org/docs/latest/ml-pipeline.html)
Overview: estimators, transformers and pipelines（概览：评估器，转换器和管道）
Extracting, transforming and selecting features（抽取，转换和选取特征）
Classification and regression（分类和回归）
Clustering（聚类）
Advanced topics（高级主题）
