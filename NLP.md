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
- **Negative Sampling**：in terms of skip-gram model with negative sampling, train binary logistic regressions for a true	pair (center word and word in its context window) versus a couple of noise pairs (the center word paired with a random word)
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
- TF-IDF：一个容易想到的思路，就是找到出现次数最多的词。如果某个词很重要，它应该在这篇文章中多次出现。于是，我们进行"词频"（Term Frequency，缩写为TF）统计。出现次数最多的词是----"的"、"是"、"在"----这一类最常用的词。它们叫做"停用词"（stop words），表示对找到结果毫无帮助、必须过滤掉的词。用统计学语言表达，就是在词频的基础上，要对每个词分配一个"重要性"权重。最常见的词（"的"、"是"、"在"）给予最小的权重，较常见的词（"中国"）给予较小的权重，较少见的词（"蜜蜂"、"养殖"）给予较大的权重。这个权重叫做逆文档频率"（Inverse Document Frequency，缩写为IDF），它的大小与一个词的常见程度成反比。
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