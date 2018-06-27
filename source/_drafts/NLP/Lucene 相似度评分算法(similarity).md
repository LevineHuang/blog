## ES相似度评分原理

Lucene（或 ElasticSearch）使用 [布尔模型（Boolean model）](http://en.wikipedia.org/wiki/Standard_Boolean_model) 查找匹配文档， 并用一个名为 [实用评分函数（practical scoring function](https://www.elastic.co/guide/cn/elasticsearch/guide/current/practical-scoring-function.html) 的公式来计算相关度。这个公式借鉴了 [词频/逆向文档频率（term frequency/inverse document frequency）](http://en.wikipedia.org/wiki/Tfidf) 和 [向量空间模型（vector space model）](http://en.wikipedia.org/wiki/Vector_space_model)，同时也加入了一些现代的新特性，如协调因子（coordination factor），字段长度归一化（field length normalization），以及词或查询语句权重提升。

### 布尔模型

*布尔模型（Boolean Model）* 只是在查询中使用 `AND` 、 `OR` 和 `NOT` （与、或和非）这样的条件来查找匹配的文档，以下查询：

```
full AND text AND search AND (elasticsearch OR lucene)
```

会将所有包括词 `full` 、 `text` 和 `search` ，以及 `elasticsearch` 或 `lucene` 的文档作为结果集。

这个过程简单且快速，它将所有可能不匹配的文档排除在外。



### Lucene 相似度评分算法(similarity)

- TF/IDF (词频/逆文档频率)算法，ES5.0之前，TF/IDF是默认的评分算法,TF/IDF源于**向量空间模型(Vector Space Model)**

- BM25算法，ES5.0及之后（2017-05-04发布的5.4版本），BM25是默认的评分算法,BM25源于**概率相关模型(probabilistic relevance model)**

  

#### TF/IDF (词频/逆文档频率)算法

根据分词词库，所有的文档在建立索引的时候进行分词划分。进行搜索的时候，也对搜索的短语进行分词划分。 搜索短语的每个分词项和每个索引中的文档根据TF/IDF进行词频出现的评分计算。 然后每个分词项的得分相加，就是这个搜索对应的文档得分。 
$$
Score(q,d) = coord(q,d) · queryNorm(q) · \sum_{t\ in\ q} (tf(t\ in\ d))·idf(t)^2·t.getBoost() · norm(t,d)
$$
这个评分公式有6个部分组成

- coord(q,d) 评分因子，基于文档中出现查询项的个数。越多的查询项在一个文档中，说明文档的匹配程度越高。
- queryNorm(q),这个因素对所有文档都是一样的值，所以它不影响排序结果。 
- tf(t in d) 指项t在文档d中出现的次数frequency。具体值为次数的开根号。
- idf(t) 反转文档频率, 出现项t的文档数docFreq。
- t.getBoost 查询时候查询项加权。
- norm(t,d) 长度相关的加权因子，目的是为了将同样匹配的文档，比较短的放比较前面 。

#### BM25算法

BM25(Best Matching)是在信息检索系统中根据提出的query对document进行评分的算法。 其主要思想是：对Query进行语素解析，生成语素$q_i$；然后，对于每个搜索结果d，计算每个语素$q_i$与d的相关性得分，最后，将$q_i$相对于d的相关性得分进行加权求和，从而得到Query与d的相关性得分。

BM25算法的一般性公式如下：
$$
Score(Q,d) = \sum W_i · R(q_i,d)
$$
其中，Q表示Query，$q_i$表示Q解析之后的一个语素（对中文而言，我们可以把对Query的分词作为语素分析，每个词看s成语素$q_i$）；d表示一个搜索结果文档；$W_i$表示语素$q_i$的权重；$R(q_i，d)$表示语素$q_i$与文档d的相关性得分。

下面我们来看如何定义$W_i$。判断一个词与一个文档的相关性的权重，方法有多种，较常用的是IDF。这里以IDF为例，公式如下：
$$
IDF(q_i) = log \frac {N-n(q_i)+0,5}{n(q_i)+0.5}
$$
其中，N为索引中的全部文档数，$n(q_i)$为包含了$q_i$的文档数。

根据IDF的定义可以看出，对于给定的文档集合，包含了$q_i$的文档数越多，$q_i$的权重则越低。也就是说，当很多文档都包含了$q_i$时，$q_i$的区分度就不高，因此使用$q_i$来判断相关性时的重要度就较低。

再来看语素$q_i$与文档d的相关性得分$R(q_i，d)$。首先来看BM25中相关性得分的一般形式：


$$
R(q_i,d) = \frac{f_i·(k_i+1)}{f_i+K} · \frac{qf_i·(k_2+1)}{qf_i+k_2}
$$

$$
K = k_1·(1-b+b·\frac{l_d}{l_{avg}})
$$

其中，k1，k2，b为调节因子，通常根据经验设置，一般k1=2，b=0.75；$f_i$为$q_i$在d中的出现频率，$qf_i$为$q_i$在Query中的出现频率。$l_d$为文档d的长度，$l_{avg}$为所有文档的平均长度。由于绝大部分情况下，$q_i$在Query中只会出现一次，即$qf_i$=1，因此公式可以简化为：
$$
R(q_i,d) = \frac{f_i·(k_1+1)}{f_i+K}
$$
从K的定义中可以看到，参数b的作用是调整文档长度对相关性影响的大小。b越大，文档长度的对相关性得分的影响越大，反之越小。而文档的相对长度越长，K值将越大，则相关性得分会越小。这可以理解为，当文档较长时，包含qi的机会越大，因此，同等fi的情况下，长文档与qi的相关性应该比短文档与qi的相关性弱。

综上，BM25算法的相关性得分公式可总结为：
$$
Score(Q,d) = \sum_i^nIDF(q_i)·\frac{f_i·(k_i+1)}{f_i+k_1·(1-b+b·\frac{l_d}{l_{avg}})}
$$
从BM25的公式可以看到，通过使用不同的语素分析方法、语素权重判定方法，以及语素与文档的相关性判定方法，我们可以衍生出不同的搜索相关性得分计算方法，这就为我们设计算法提供了较大的灵活性。

 

### Reference

[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)

 

 