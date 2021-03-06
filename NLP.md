# 自然语言处理
- **句法语义分析**：对于给定的句子，进行分词、词性标记、命名实体识别和链接、句法分析、语义角色识别和多义词消歧。
- **信息抽取**：从给定文本中抽取重要的信息，比如，时间、地点、人物、事件、原因、结果、数字、日期、货币、专有名词等等。通俗说来，就是要了解谁在什么时候、什么原因、对谁、做了什么事、有什么结果。涉及到实体识别、时间抽取、因果关系抽取等关键技术。
- **文本挖掘（或者文本数据挖掘）**：包括文本聚类、分类、信息抽取、摘要、情感分析以及对挖掘的信息和知识的可视化、交互式的表达界面。目前主流的技术都是基于统计机器学习的。
- **机器翻译**：把输入的源语言文本通过自动翻译获得另外一种语言的文本。根据输入媒介不同，可以细分为文本翻译、语音翻译、手语翻译、图形翻译等。机器翻译从最早的基于规则的方法到二十年前的基于统计的方法，再到今天的基于神经网络（编码-解码）的方法，逐渐形成了一套比较严谨的方法体系。
- **信息检索**：对大规模的文档进行索引。可简单对文档中的词汇，赋之以不同的权重来建立索引，也可利用1，2，3的技术来建立更加深层的索引。在查询的时候，对输入的查询表达式比如一个检索词或者一个句子进行分析，然后在索引里面查找匹配的候选文档，再根据一个排序机制把候选文档排序，最后输出排序得分最高的文档。
- **问答系统**：对一个自然语言表达的问题，由问答系统给出一个精准的答案。需要对自然语言查询语句进行某种程度的语义分析，包括实体链接、关系识别，形成逻辑表达式，然后到知识库中查找可能的候选答案并通过一个排序机制找出最佳的答案。
- **对话系统**：系统通过一系列的对话，跟用户进行聊天、回答、完成某一项任务。涉及到用户意图理解、通用聊天引擎、问答引擎、对话管理等技术。此外，为了体现上下文相关，要具备多轮对话能力。同时，为了体现个性化，要开发用户画像以及基于用户画像的个性化回复。
- **word embedding**：首先我们解释一下词嵌入(word embedding）的概念。咱们之前的所有向量表示都是稀疏的，通常都是一个高维的向量，向量里面的元素大部分都是0。那么embedding有什么不一样的呢？Embedding 同样也是用一个向量来表示一个词，但是它是使用一个较低维度、稠密地表示(每一维度都有非零的数值)。word embedding有好处：
    - 不会造成维度爆炸（curse of dimensionality），因为维度是我们自己设置的，通常比较小
    - 向量是稠密的，不需要稀疏向量所采用的各种优化算法来提升计算效率
- **word2vec**：其实就是把单词表示成固定维度的稠密的向量。word2vec 有两种常用的数据准备方式：CBOW (continuous bag of words)，用前后词（context words）预测目标词（target word）。skip-gram，用目标词（target word）预测前后词（context word）。word2vec 并不关心相邻单词之前一起出现的频数，而是仅仅关心，这个单词是不是属于另一个单词的上下文(context)!也就是说，word2vec 不关心根据这个词预测出的下一个词语是什么，而是只关心这两个词语之间是不是有上下文关系。于是，word2vec 需要的仅仅是一个二分类器：“这个单词是另一个单词的上下文单词吗？”
所以，要训练一个 word2vec模型，我们其实是在训练一个二分类器。而二分类器，你肯定很容易就想到了 Logistic Regression。实际情况，skip-gram 用的比较多，因为有一个说法，CBOW 模型在小的数据集上面表现不错，在大的数据集里，skip-gram表现更好。word2vec的输出层是softmax，因此对于CBOW，则输出各个word是target word的概率，结合训练集的标注便可以训练这些输入的context word的weights，而这些weights，就是输入词的vector。同理，skip-gram输出层输出的是各个word是它context word的概率，此时weight向量就是输入词的向量。因此，word2vec可以平均这两个模型输出的结果，得到更准确的vector。
- **BLEU**：BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
- **Negative Sampling**：in terms of skip-gram model with negative sampling, train binary logistic regressions for a true	pair (center word and word in its context window) versus a couple of noise pairs (the center word paired with a random word) 负采样可以加速损失函数的减少，从而加速训练速度。负采样主要用于正样本远小于负样本的情况，因此在训练中提供的样本大概率是负样本，因此将损失函数里负样本的数量只需要采样几个负样本计算它们的概率分布来代表所有的负样本的概率分布。
- 在实践中，如果领域的数据非常少，我们可能直接用在其它任务中Pretraining 的Embedding并且fix住它；而如果领域数据较多的时候我们会用Pretraining 的Embedding 作为初始值，然后用领域数据驱动它进行微调。
- **Language Model**：语言模型就是指一个模型能基于之前出现的序列信息去预测下一个可能生成的信息。
- **准确率p**：预测对的数 / 预测出来的总数
- **召回率r**：预测出来的对的数 / 客观正确的数目
- **F1 Score**：2pr / (p+r)
- **Attention Mechanism**：Weighted combination of all the input states. With an attention mechanism we no longer try encode the full source sentence into a fixed-length vector. Rather, we allow the decoder to “attend” to different parts of the source sentence at each step of the output generation. Importantly, we let the model learn what to attend to based on the input sentence and what it has produced so far. So, in languages that are pretty well aligned (like English and German) the decoder would probably choose to attend to things sequentially. Attending to the first word when producing the first English word, and so on
- NLP一共有四大类任务：
    - 序列标注：分词，词性标注，命名实体识别
    - 分类任务：文本分类，情感分析
    - 句子关系判断：自然语言推理，深度文本匹配，问答系统
    - 生成式任务：机器翻译，文本摘要生成
- **[BiLSTM](https://zhuanlan.zhihu.com/p/47802053)**
- [NLP中的深度学习技术](https://zhuanlan.zhihu.com/p/57979184)
- **Cos Similarity VS Dot Product**: Cosine similarity only cares about angle difference, while dot product cares about angle and magnitude. If you normalize your data to have the same magnitude, the two are indistinguishable. Sometimes it is desirable to ignore the magnitude, hence cosine similarity is nice, but if magnitude plays a role, dot product would be better as a similarity measure. Note that neither of them is a "distance metric".

## NLP Pipeline
1. **Sentence Segmentation**：断句，句子切分
2. **Tokenization**：分词，中文和英文不一样，中文可能是两个字为一个词语，而英文是词和词之间会用空格隔开
3. **Part of Speech Tagging**：词性标注，我们需要区分出一个词在句子中的角色，是名词？动词？还是介词。我们使用一个预先经过几百万英文句子训练、被调教好的词性标注（POS: Part Of Speech）分类模型，模型只是基于统计结果给词打上标签，它并不了解一个词的真实含义，这一点和人类对词语的理解方式是完全不同的
4. **Lemmatization**：文本词形还原，很多基于字母拼写的语言，像英语、法语、德语等，都会有一些词形的变化，比如单复数变化、时态变化等
5. **Stop Words**：去停用词，在信息检索中，为节省存储空间和提高搜索效率，在处理自然语言数据（或文本）之前或之后会自动过滤掉某些字或词，这些字或词即被称为Stop Words(停用词)。这些停用词都是人工输入、非自动化生成的，生成后的停用词会形成一个停用词表。但是，并没有一个明确的停用词表能够适用于所有的工具。甚至有一些工具是明确地避免使用停用词来支持短语搜索的。仅仅是起到衔接和辅助表述的作用。他们的存在，对计算机来说更多是噪音。所以我们需要把这些词识别出来。现在虽然停用词列表很多，但一定要根据实际情况进行配置。比如英语的the，通常情况是停用词，但很多乐队名字里有the这个词，The Doors, The Who，甚至有个乐队直接就叫The The！这个时候就不能看做是停用词了。
6. **Dependency Parsing**：解析依赖关系，解析句子中每个词之间的依赖关系，最终建立起一个关系依赖树。这个树的root是关键动词，从这个关键动词开始，把整个句子中的词都联系起来。
7. **Noun Phrases**：我们还可以选择把相关的词进行合并分组，例如把名词以及修饰它的形容词合并成一个词组短语。不过这一步工作不是必须要有的，视具体情况而定。
8. **Named Entity Recognition**：命名实体识别，识别出那些具有特殊属性的名字，比如国家、地理位置、时间和人名等
9. **Coreference Resolution**：共指消解，人类的语言很复杂，但在使用过程中却是倾向于简化和省略的。比如他，它，这个，那个，前者，后者…这种指代的词，再比如缩写简称，北京大学通常称为北大，中华人民共和国通常就叫中国。这种现象，被称为共指现象。在特定语境下人类可以毫不费力的区别出它这个字，到底指的是牛，还是手机。但是计算机需要通过共指消解才能知道下面这句话

## Information Retrieval
- TF-IDF：一个容易想到的思路，就是找到出现次数最多的词。如果某个词很重要，它应该在这篇文章中多次出现。于是，我们进行"词频"（Term Frequency，缩写为TF）统计。出现次数最多的词是----"的"、"是"、"在"----这一类最常用的词。它们叫做"停用词"（stop words），表示对找到结果毫无帮助、必须过滤掉的词。用统计学语言表达，就是在词频的基础上，要对每个词分配一个"重要性"权重。最常见的词（"的"、"是"、"在"）给予最小的权重，较常见的词（"中国"）给予较小的权重，较少见的词（"蜜蜂"、"养殖"）给予较大的权重。这个权重叫做逆文档频率"（Inverse Document Frequency，缩写为IDF），它的大小与一个词的常见程度成反比。TF-IDF的优点是简单快速，而且容易理解。缺点是有时候用词频来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。如果要体现词的上下文结构，那么你可能需要使用word2vec算法来支持。另外的就是，TF-IDF中文档是用weighted Bag of words来表示，这里的weighted就是指TF和IDF的乘积，文档由该文档中出现的词来表示，优点是简单有效，缺点是无法从词袋表示来恢复原文档以及它忽略了词之间和句法关系以及篇章结构的信息。
```
TF = 某个词在文章中出现的次数 / 文章的总词数
IDF = log(语料库的文档总数 / 包含该词的文档数+1)
TF-IDF = TF * IDF
```
- [BM25](https://www.jianshu.com/p/1e498888f505)
- **Semanctic Matching**: conduct query/document analysis to represent the meanings of query/document with richer representations and then perform matching with the representations.
- **Text Matching as supervised objective**:
    - **Representation-based models**: Representation-based models construct a fixed-dimensional vector representation for each text separately and then perform matching within the latent space. (DSSM, CDSSM, ARC-I)
    - **Interaction-based models**: Interaction-based models compute the interaction between each individual term of both texts. An interaction can be identity or syntactic/semantic similarity. The interaction matrix is subsequently summarized into a matching score. (DRMM, MatchPyramid, Match-SRNN, K-NRM)
    - **Hybrid models**: Hybrid models consist of (i) a representation component that combines a sequence of words (e.g., a whole text, a window of words) into a fixed-dimensional representation and (ii) an interaction component. These two components can occur (1) in serial or (2) in parallel. (ARC-II, MV-LSTM, Duet, DeepRank)
- [**信息检索评价**](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))：衡量检索结果与标准答案的一致性
    - 对 Unranked Retrieval(非排序检索)的评价
        - P@k (precision at k)
        - R@k (Recall at k)
        - F1
    - 对 Ranked Retrieval(排序结果)的评价，考虑相关文档在检索结果中的排序位置，考虑在不同 recall levels 的 precision 值
        - AP and MAP
        - [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
        - nDCG (Normalized Discounted Cumulative Gain)
- **[How Lucene does indexing](https://stackoverflow.com/questions/2602253/how-does-lucene-index-documents)** 
- **语义匹配**：例如在搜索中，同样想知道iPhone手机的价格，两个query:“iphone多少钱”和“苹果手机什么价格”，这两个query的意思是完全一样的，但是字面上没有任何的重叠，用bm25和tfidf来计算，他们的相似度都是0。语义匹配就是要解决这种字面上不一定重叠，但是语义层面是相似的文本匹配问题。
- **DSSM**: 
![](./figures/DSSM.png)
![](./figures/DSSM_illustration.png)
    - latent semantic models with a deep structure that project queries and documents into a common low-dimensional space
    - relevance of a document given a query is readily computed as the distance between them
    - word hashing method, through which the high-dimensional term vectors of queries or documents are projected to low-dimensional letter based n-gram vectors with little information loss
    - The input (raw text features) to the DNN is a highdimensional term vector, e.g., raw counts of terms in a query or a document without normalization, and the output of the DNN is a concept vector in a low-dimensional semantic feature space
    - Compared with the original size of the one-hot vector, word hashing allows us to represent a query or a document using a vector with much lower dimensionality. Take the 40Kword vocabulary as an example. Each word can be represented by a 10,306-dimentional vector using letter trigrams, giving a fourfold dimensionality reduction with few collisions. For instance, "good" >>> [0,0,0,...1,0,0,...] with size of 500K ===>> #go, goo, ood, od# >>> [0,1,1,...1,....] with size of 30,621
    - 使用bag of letter-trigrams的好处：
        - 减少字典的大小，#words 500K -> letter-trigram: 30K
        - 处理out of vocabulary的问题，对于训练数据中没出现的单词也可以处理，提高了泛化性
        - 对于拼写错误也有一定的鲁棒性
    - DSSM的缺点：
        - 词袋模型，失去了词序信息
        - point-wise的loss（相关文档为1,不相关文档为0,而不是一个匹配程度），和排序任务match度不够高。可以轻松的扩展到pair-wise的loss，这样和排序任务更相关。
- **获取词序信息**：
    - CNN
![](./figures/CNN.png)
    - RNN
![](figures/RNN.png)
- **C-DSSM**: 
![](./figures/C-DSSM.png)
    - C-DSSM has a convolutional layer that projects each word within a context window to a local contextual feature vector. Semantically similar words-withincontext are projected to vectors that are close to each other in the contextual feature space.
    - the overall semantic meaning of a sentence is often determined by a few key words in the sentence, thus, simply mixing all words together (e.g., by summing over all local feature vectors) may introduce unnecessary divergence and hurt the effectiveness of the overall semantic representation. Therefore, C-DSSM uses a max pooling layer to extract the most salient local features to form a fixed-length global feature vector.
- **ARC-I**:
![](./figures/ARC-1.png)
![](./figures/ARC-2.png)
    - we devise novel deep convolutional network architectures that can naturally combine 1) the hierarchical sentence modeling through layer-by-layer composition and pooling, and 2) the capturing of the rich matching patterns at different levels of abstraction
![](./figures/ARC-3.png)
    -  Although ARC-I enjoys the flexibility brought by the convolutional sentence model, it suffers from a drawback inherited from the Siamese architecture: it defers the interaction between two sentences (in the final MLP) to until their individual representation matures (in the convolution model), therefore runs at the risk of losing details (e.g., a city name) important for the matching task in representing the sentences
![](figures/ARC-4.png)
![](./figures/ARC-5.png)
    - In view of the drawback of Architecture-I, we propose Architecture-II (ARC-II) that is built directly on the interaction space between two sentences. It has the desirable property of letting two sentences meet before their own high-level representations mature, while still retaining the space for the individual development of abstraction of each sentence
- **[搜索与推荐中的深度学习匹配](https://zhuanlan.zhihu.com/p/38296950)**
- **Ad-hoc retrieval**: Ad-hoc retrieval is a classic retrieval task in which the user specifies his/her information need through a query which initiates a search (executed by the information system) for documents that are likely to be relevant to the user.
    - A major characteristic of ad-hoc retrieval is the heterogeneity of the query and the documents.
    - The query comes from a search user with potentially unclear intent and is usually very short, ranging from a few words to a few sentences.
    - The documents are typically from a different set of authors and have longer text length, ranging from multiple sentences to many paragraphs. Such heterogeneity leads to the **critical vocabulary mismatch problem**. Vocabulary mismatch means the terms between the questions and relevant documents are different but having the same semantic and these relevant documents cannot be retrieved successfully.
    -  Semantic matching, matching words and phrases with similar meanings, could alleviate the problem, but exact matching is indispensable especially with rare terms.
- **[Learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank)**
    - Learning to rank[1] or machine-learned ranking (MLR) is the application of machine learning, typically supervised, semi-supervised or reinforcement learning, in the construction of ranking models for information retrieval systems.[2]
    - The ranking model's purpose is to rank, i.e. produce a permutation of items in new, unseen lists in a way which is "similar" to rankings in the training data in some sense.
    - Training data consists of queries and documents matching them together with relevance degree of each match. Training data is used by a learning algorithm to produce a ranking model which computes the relevance of documents for actual queries.
    - Typically, users expect a search query to complete in a short time (such as a few hundred milliseconds for web search), which makes it impossible to evaluate a complex ranking model on each document in the corpus, and so a two-phase scheme is used.[5] First, a small number of potentially relevant documents are identified using simpler retrieval models which permit fast query evaluation, such as the vector space model, boolean model, weighted AND,[6] or BM25. This phase is called top-k document retrieval and many heuristics were proposed in the literature to accelerate it, such as using a document's static quality score and tiered indexes.[7] In the second phase, a more accurate but computationally expensive machine-learned model is used to re-rank these documents.
- **Index in IR**: Inverted index consists of indexing terms and indexing documents. The terms can be different forms, such as a single word or continuous two words. Also, indexing documents can be different granularities, such as section, paragraph or even sentence. In addition, indexing documents can be accompanied with some other useful information, such as term frequency.

### Hyperparameter Optimization
- In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.
- Grid search: The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set[3] or evaluation on a held-out validation set. Since the parameter space of a machine learner may include real-valued or unbounded value spaces for certain parameters, manually set bounds and discretization may be necessary before applying grid search.
- Random search: Random Search replaces the exhaustive enumeration of all combinations by selecting them randomly. This can be simply applied to the discrete setting described above, but also generalizes to continuous and mixed spaces.
- Bayesian optimization
- Gradient-based optimization
- Evolutionary optimization
- Population-based