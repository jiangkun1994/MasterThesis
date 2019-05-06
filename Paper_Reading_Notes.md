## Paper Reading Notes
### [Passage Re-Ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf)
- A simple question answering pipeline:
    - A large number (e.g., a thousand) of possibily relevant documents to a given question retrieved by a standard mechanism (e.g, BM25).
    - Each of these documents is scored and re-ranked by a more computaionally-intensive method. (passage re-ranking)
    - Top ten of fifty of these documents will be the source for candidate answer by an answer generation module.
- truncate query and passage text
- [CLS] token is inserted at the beginning of the first sentence.
- BERT (large) is used as a binary classification, then predict the probability of each passage.

### [Learning to Transform, Combine, and Reason in Open-Domain Question Answering](https://e.humanities.uva.nl/publications/2019/dehg_lear19.pdf)
- The most important point of this paper is to look for a better way to answer complex questions given a set of documents (no need to remove noisy information).
- Learning to attend on essential terms: 
    - Query reformulations to improve retrieval performance.
    - Reader can distinguishes essential terms and distracting terms.

### [Nerual Reading Comprehension and Beyond](https://purl.stanford.edu/gd576xb1833)
- Motivation (why do research on neural reading comprehension):
    - What it means to understand human language
    - NLP community puts efforts on teaching computer to understand human language on various aspects
    - How to evaluate whether a model can understand language and capture these aspects of text
    - Using reading comprehension is a good evaluation
    - Why neural reading comprehension? (Has rapid progress)
    - If had a high-performing RC system, it will be beneficial to many practical applications
    - Two research directions involved with RC: open-domain QA and conversational QA
- neural network models are better at recognizing lexical matches and paraphrases compared to conventional feature-based classifiers.
- existing reading comprehension tasks into four categories: (1) Cloze style (2) Multiple choice (3) Span prediction (4) Free-form answer
- Reading comprehension vs. question answering




## For Thesis Writing
### [Adaptive Document Retrieval for Deep Question Answering](https://arxiv.org/pdf/1808.06528.pdf)
- Having the metric Answer Recall
- State-of-the-art systems in deep question answering proceed as follows: (1) an initial document retrieval selects relevant documents, which (2) are then processed by a neural network in order to extract the final answer. Yet the exact interplay between both components is poorly understood, especially concerning the number of candidate documents that should be retrieved. We show that choosing a static number of documents – as used in prior research – suffers from a noiseinformation trade-off and yields suboptimal results.
- Question-answering (QA) systems proceed by following a two-staged process (Belkin, 1993 Nicholas Belkin. 1993. Interaction with texts: Information retrieval as information-seeking behavior): in a first step, a module for document retrieval selects n potentially relevant documents from a given corpus. Subsequently, a machine comprehension module extracts the final answer from the previously-selected documents.
- The latter step often involves hand-written rules or machine learning classifiers (c. f. Shen and Klakow, 2006; Kaisser and Becker, 2004), and recently also deep neural networks (e. g. Chen et al., 2017; Wang et al., 2018)
-  A larger n improves the recall of document retrieval and thus the chance of including the relevant information. However, this also increases the noise and might adversely reduce the accuracy of answer extraction

### [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/pdf/1902.01718v1.pdf)
- Having the metric Answer Recall
- Using BM25
- In contrast, what we refer to as “end-to-end” question answering begins with a large corpus of documents. Since it is impractical to apply inference exhaustively to all documents in a corpus with current models (mostly based on neural networks), this formulation necessarily requires some type of term-based retrieval technique to restrict the input text under consideration—and hence an architecture quite like systems from over a decade ago
- As NLP researchers became increasingly interested in QA, they placed greater emphasis
on the later stages of the pipeline to emphasize
various aspects of linguistic analysis. Information retrieval techniques receded into the background and became altogether ignored. Most popular QA benchmark datasets today—for example,
TrecQA (Yao et al., 2013), WikiQA (Yang et al.,
2015), and MSMARCO (Bajaj et al., 2016)—
are best characterized as answer selection tasks.
That is, the system is given the question as well
as a candidate list of sentences to choose from.
Similarly, reading comprehension datasets such as
SQuAD (Rajpurkar et al., 2016) eschew retrieval
entirely, since there is only a single document from
which to extract answers.

### [$R^3$: Reinforced Ranker-Reader for Open-Domain Question Answering](https://arxiv.org/pdf/1709.00023.pdf)
- Having Answer Recall
- Using BM25+TFIDF
- In the QA community, “openness” can be interpreted as referring either to the scope of question topics or to the breadth and generality of the knowledge source used to answer each question. Following (Chen et al. 2017a) we adopt the latter definition.
- Recent deep learning-based research has focused on open-domain QA based on large text corpora such as Wikipedia, applying information retrieval (IR) to select passages and reading comprehension (RC) to extract answer phrases (Chen et al. 2017a; Dhingra, Mazaitis, and Cohen 2017)
- Dividing the pipeline into IR and RC stages leverages an enormous body of research in both IR and RC, including recent successes in RC via neural network techniques (Wang and Jiang 2017b; Wang et al. 2016; Xiong, Zhong, and Socher 2017; Wang et al. 2017).
- The main difference between training SR-QA and standard RC models is in the passages used for training. In standard RC model training, passages are manually selected to guarantee that ground-truth answers are contained and annotated within the passage (Rajpurkar et al. 2016).

### [Ranking Paragraphs for Improving Answer Recall in Open-Domain Question Answering](https://aclweb.org/anthology/D18-1053)
- Using TFIDF and then adding a re-ranker
- As open-domain QA requires retrieving relevant documents from text corpora to answer questions, its performance largely depends on the performance of document retrievers. However, since traditional information retrieval systems are not effective in obtaining documents with a high probability of containing answers, they lower the performance of QA systems
- With the introduction of large scale machine comprehension datasets, machine comprehension models that are highly accurate and efficient in answering questions given raw texts have been proposed recently (Seo et al., 2016; Xiong et al., 2016; Wang et al., 2017c).
- While conventional machine comprehension models were given a paragraph that always contains an answer to a question, some researchers have extended the models to an open-domain setting where relevant documents have to be searched from an extremely large knowledge source such as Wikipedia (Chen et al., 2017; Wang et al., 2017a).
- However, most of the open-domain QA pipelines depend on traditional information retrieval systems

### [Retrieve-and-Read: Multi-task Learning of Information Retrieval and Reading Comprehension](https://arxiv.org/pdf/1808.10628.pdf)
- Using TF-IDF and adding a neural IR model
- This study considers the task of machine reading at scale (MRS)
wherein, given a question, a system first performs the information retrieval (IR) task of finding relevant passages in a knowledge
source and then carries out the reading comprehension (RC) task
of extracting an answer span from the passages
- Large and highquality datasets that are sufficient to train deep neural networks
have been constructed; in particular, the SQuAD dataset [38] has brought significant progress such that the RC performance of AI
is now comparable to that of humans
- Chen et al. proposed DrQA, which is an opendomain QA system using Wikipedia’s texts as a knowledge source
by simply combining an exact-matching IR method with an RC
method based on a neural network [5].
- Their system showed promising results; however, the results indicated that the IR method, which
retrieved the top five passages from five million articles for each
question, was a bottleneck in terms of accuracy. It can retrieve passages that contain question words, but such passages are not always relevant to the question.
- Although our neural model can alleviate the bottleneck of IR
accuracy, adapting it to the whole of a large-scale corpus causes
computational complexity problems.
- We therefore introduce telescoping settings [27], where our IR model component re-ranks the
outputs of fast exact-matching models that focus on eliminating
higher irrelevant passages (Figure 1). This idea enables our model
to perform at a practical speed without loss of generality
- Without loss of generality, we can use a telescoping setting with
our model, where our IR component finds relevant passages in a
subset of a corpus D retrieved by chaining of different IR models.
That is, the initial rankers focus on eliminating higher irrelevant
passages, and our model operates as a re-ranker for determining
the existence of answer phrases within the remaining passages. 
- We used Document
Retriever [5], which is a model based on bigram hashing and TFIDF matching, for both the first and second retrievals. Finally, the
IR component of our model found the top-1 passage from the 200
passages and passed it to our RC component.

### [HAS-QA: Hierarchical Answer Spans Model for Open-domain Question Answering](https://arxiv.org/pdf/1901.03866v1.pdf)
- Using search engine
- Open-domain question answering (OpenQA) aims to seek
answers for a broad range of questions from a large knowledge sources, e.g., structured knowledge bases (Berant et
al. 2013; Mou et al. 2017) and unstructured documents from
search engine (Ferrucci et al. 2010). In this paper we focus on the OpenQA task with the unstructured knowledge
sources retrieved by search engine.
- The answer string is a piece of text that can answer the question. If the answer string is obtained in a paragraph as a consecutive
text, we call it the answer span.
- RC task assumes that the given paragraph contains
the answer string (Figure 1 top), however, it is not valid for the OpenQA task (Figure 1 bottom). That’s because the
paragraphs to provide answer for an OpenQA question is
collected from a search engine, where each retrieved paragraph is merely relevant to the question
- DrQA (Chen et al. 2017) is the earliest work that applies RC
model in OpenQA task.

### [LEARNING TO ATTEND ON ESSENTIAL TERMS: AN ENHANCED RETRIEVER-READER MODEL FOR OPENDOMAIN QUESTION ANSWERING](https://arxiv.org/pdf/1808.09492v4.pdf)
- However, existing techniques struggle to retrieve indirectly related
evidence when no directly related evidence is provided, especially for complex
questions where it is hard to parse precisely what the question asks. 
- However, this assumption ignores the difficulty
of retrieving question-related evidence from a large volume of open-domain resources, especially
when considering complex questions which require reasoning or commonsense knowledge.

### [Training a Ranking Function for Open-Domain Question Answering](https://arxiv.org/pdf/1804.04264v1.pdf)
- Only build neural reranker as the dataset QUASAR-T has already provided only 100 passages retrieved by search engine
- Recently, the stateof-the-art machine reading models achieve human level performance in SQuAD which is
a reading comprehension-style question answering (QA) task. The success of machine
reading has inspired researchers to combine
information retrieval with machine reading
to tackle open-domain QA. However, these
systems perform poorly compared to reading
comprehension-style QA because it is difficult
to retrieve the pieces of paragraphs that contain the answer to the question
- In reading comprehension-style
QA, the ground truth paragraph that contains the
answer is given to the system whereas no such information is available in open-domain QA setting.
- Open-domain QA systems have generally been
built upon large-scale structured knowledge bases,
such as Freebase or DBpedia. The drawback of
this approach is that these knowledge bases are not
complete (West et al., 2014), and are expensive to
construct and maintain.
- Another method for open-domain QA is a
corpus-based approach where the QA system
looks for the answer in the unstructured text corpus (Brill et al., 2001). This approach eliminates
the need to build and update knowledge bases by
taking advantage of the large amount of text data
available on the web
- As machine
readers are excellent at this task, there have been
attempts to combine search engines with machine
reading for corpus-based open-domain QA (Chen
et al., 2017; Wang et al., 2017).
- To achieve high
accuracy in this setting, the top documents retrieved by the search engine must be relevant to
the question. As the top ranked documents returned from search engine might not contain the
answer that the machine reader is looking for, reranking the documents based on the likelihood of
containing answer will improve the overall QA
performance
- Semantic similarity is crucial in QA as the passage containing the answer may be semantically
similar to the question but may not contain the exact same words in the question. For example, the
answer to “What country did world cup 1998 take
place in?” can be found in “World cup 1998 was
held in France.”
- In machine reading-style question answering datasets like SQuAD, the system has to locate the answer to a question in the given ground truth
paragraph. Neural network based models excel at
this task and have recently achieved human level
accuracy in SQuAD.
- Following the advances in machine reading,
researchers have begun to apply Deep Learning in corpus-based open-domain QA approach
by incorporating information retrieval and machine reading. Chen et al. (2017) propose a QA
pipeline named DrQA that consists of a Document Retriever and a Document Reader. The
Document Retriever is a TF-IDF retrieval system built upon Wikipedia corpus. The Document
Reader is a neural network machine reader trained
on SQuAD. Although DrQA’s Document Reader
achieves the exact match accuracy of 69.5 in reading comprehension-style QA setting of SQuAD,
their accuracy drops to 27.1 in the open-domain
setting, when the paragraph containing the answer
is not given to the reader
-  In order to extract the
correct answer, the system should have an effective retrieval system that can retrieve highly relevant paragraphs. Therefore, retrieval plays an important role in open-domain QA and current systems are not good at it

### [EVIDENCE AGGREGATION FOR ANSWER RE-RANKING IN OPEN-DOMAIN QUESTION ANSWERING](https://arxiv.org/pdf/1711.05116v2.pdf)
- using a search engine such as Google or Bing.
- Open-domain question answering (QA) aims to answer questions from a broad range of domains
by effectively marshalling evidence from large open-domain knowledge sources. Such resources
can be Wikipedia (Chen et al., 2017), the whole web (Ferrucci et al., 2010), structured knowledge
bases (Berant et al., 2013; Yu et al., 2017) or combinations of the above (Baudis &ˇ Sediv ˇ y, 2015). Recent work on open-domain QA has focused on using unstructured text retrieved from the web
to build machine comprehension models (Chen et al., 2017; Dhingra et al., 2017b; Wang et al.,
2017).
- We conduct experiments on three publicly available open-domain QA datasets, namely, QuasarT (Dhingra et al., 2017b), SearchQA (Dunn et al., 2017) and TriviaQA (Joshi et al., 2017). These
datasets contain passages retrieved for all questions using a search engine such as Google or Bing.
We do not retrieve more passages but use the provided passages only.

### [PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text](https://arxiv.org/pdf/1904.09537v1.pdf)
- Open domain Question Answering (QA) is the
task of finding answers to questions posed in natural language, usually using text from a corpus
(Dhingra et al., 2017; Joshi et al., 2017; Dunn
et al., 2017), or triples from a knowledge base
(KB) (Zelle and Mooney, 1996; Zettlemoyer and
Collins, 2005; Yih et al., 2015).
-   Both of these approaches have limitations. Even the largest KBs
are incomplete (Min et al., 2013), which limits
recall of a KB-based QA system.  On the other
hand, while a large corpus may contain more answers than a KB, the diversity of natural language
makes corpus-based QA difficult (Chen et al.,
2017; Welbl et al., 2018; Kwiatkowski et al., 2019;
Yang et al., 2018).
- We focus
on tasks in which questions require compositional
(sometimes called “multi-hop”) reasoning, and a
setting in which the KB is incomplete, and hence
must be supplemented with information extracted
from text. 

### [QUASAR: DATASETS FOR QUESTION ANSWERING BY SEARCH AND READING](https://arxiv.org/pdf/1707.03904v2.pdf)
- Factoid Question Answering (QA) aims to extract
answers, from an underlying knowledge source, to information seeking questions posed in natural language. Depending on the knowledge source available there are two main approaches for factoid QA. Structured sources, including Knowledge
Bases (KBs) such as Freebase (Bollacker et al.,
2008), are easier to process automatically since
the information is organized according to a fixed
schema.
- However, even the largest KBs are often incomplete (Miller et al., 2016; West et al., 2014), and
hence can only answer a limited subset of all possible factoid questions.
- For this reason the focus is now shifting towards
unstructured sources, such as Wikipedia articles,
which hold a vast quantity of information in textual form and, in principle, can be used to answer
a much larger collection of questions.
-  Extracting
the correct answer from unstructured text is, however, challenging, and typical QA pipelines consist of the following two components: (1) searching for the passages relevant to the given question,
and (2) reading the retrieved text in order to select a span of text which best answers the question
(Chen et al., 2017; Watanabe et al., 2017).
- Machine reading performance, in particular, has been significantly boosted in the last
few years with the introduction of large-scale reading comprehension datasets such as CNN / DailyMail (Hermann et al., 2015) and Squad (Rajpurkar
et al., 2016). State-of-the-art systems for these
datasets (Dhingra et al., 2017; Seo et al., 2017) focus solely on step (2) above, in effect assuming the
relevant passage of text is already known.
-  Prior datasets (such as those used in (Chen et al.,
2017)) are constructed by first selecting a passage
and then constructing questions about that passage. This design (intentionally) ignores some of
the subproblems required to answer open-domain
questions from corpora, namely searching for passages that may contain candidate answers, and aggregating information/resolving conflicts between
candidates from many passages. 
- For the automatic systems, we see an interesting
tension between searching and reading accuracies
- retrieving more documents in the search phase leads to a higher coverage of answers, but makes
the comprehension task more difficult
-  However, the automatic procedure used to construct these questions often introduces ambiguity and makes the
task more difficult (Chen et al., 2016).
- quad in particular has attracted considerable interest,
but recent work (Weissenborn et al., 2017) suggests that answering the questions does not require
a great deal of reasoning.
- However, the documents retrieved
for TriviaQA were obtained using a commercial
search engine, making it difficult for researchers
to vary the retrieval step of the QA system in a
controlled fashion;
-  SearchQA (Dunn et al., 2017)
is another recent dataset aimed at facilitating research towards an end-to-end QA pipeline, however this too uses a commercial search engine, and
does not provide negative contexts not containing the answer, making research into the retrieval
component difficult.

### [Advances in Natural Language Question Answering: A Review](https://arxiv.org/pdf/1904.05276v1.pdf)
- Introducing the history of QA including rule-based QA systems, statistical QA systems, machine learning approaches and deep learning approaches

### [Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases](https://arxiv.org/pdf/1903.02188v2.pdf)
- Given questions
in natural language (NL), the goal of KBQA is to
automatically find answers from the underlying KB,
which provides a more natural and intuitive way to
access the vast underlying knowledge resources.
- The approaches proposed to tackle the KBQA task
can be roughly categorized into two groups: semantic parsing (SP) and information retrieval (IR) (embedding-based) approaches.

### [Improving Retrieval-Based Question Answering with Deep Inference Models](https://arxiv.org/pdf/1812.02971v1.pdf)
- Using Lucene and BM25
-  All questions are expressed solely in natural language (without diagrams or
equations) and each question has four possible answers from which only one is guaranteed to be correct. 
- Nicula et al. [8] proposed a model for predicting correct answers based on candidate
contexts extracted from Wikipedia. Using Lucene-based indexing and retrieval, each
paragraph in the English Wikipedia has been indexed and used as a candidate context
for questions and corresponding answers. Each (question, answer) pair has been
searched in the index using the standard BM25 score and the top 5 retrieved documents,
along with the question and candidate answer, are concatenated and fed into a deep
neural network that computes a score for the (question, candidate, context) triple
-  Our solution relies solely
on natural language, plain text information (we use various corpora available on the
Internet: Wikipedia, CK12 books) and not on structured knowledge at all. This is an
important advantage as plain text corpora are much easier to collect and do not require
human intervention to extract structured rules and entities
-  We use Lucene to
index a large collection of documents (entire English Wikipedia, science books collected over the Internet1
, ARC Corpus2
) pertinent for the science QA task at hand and
later to retrieve relevant information for candidate answers.
- The documents have been filtered to include only affirmative sentences without references to images or tables. The queries contain (question, candidate answer) pairs and
they are looked up in the Lucene index and ranked based on the default BM25 score
used by Lucene [13] with term boosting (一个完美的open-domain QA系统当然可以从任何一种类型的knowledge source进行信息搜索，但是我们现在只考虑plain text without referring to tables or images，as plain text corpora are much easier to collect and do not require
human intervention to extract structured rules and entities)

### [Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text](https://arxiv.org/pdf/1809.00782v1.pdf)
- Open domain Question Answering (QA) is the
task of finding answers to questions posed in natural language. Historically, this required a specialized pipeline consisting of multiple machinelearned and hand-crafted modules (Ferrucci et al.,
2010). Recently, the paradigm has shifted towards
training end-to-end deep neural network models
for the task (Chen et al., 2017; Liang et al., 2017;
Raison et al., 2018; Talmor and Berant, 2018;
Iyyer et al., 2017). Most existing models, however, answer questions using a single information
source, usually either text from an encyclopedia,
or a single knowledge base (KB).
- Intuitively, the suitability of an information
source for QA depends on both its coverage and
∗Haitian Sun and Bhuwan Dhingra contributed equally
to this work.
the difficulty of extracting answers from it. A large
text corpus has high coverage, but the information
is expressed using many different text patterns. As
a result, models which operate on these patterns
(e.g. BiDAF (Seo et al., 2017)) do not generalize
beyond their training domains (Wiese et al., 2017;
Dhingra et al., 2018) or to novel types of reasoning (Welbl et al., 2018; Talmor and Berant, 2018)
- KBs, on the other hand, suffer from low coverage due to their inevitable incompleteness and restricted schema (Min et al., 2013), but are easier
to extract answers from, since they are constructed
precisely for the purpose of being queried.
- In this paper we focus on a scenario in which a
large-scale KB (Bollacker et al., 2008; Auer et al.,
2007) and a text corpus are available, but neither
is sufficient alone for answering all questions.

### [SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine](https://arxiv.org/pdf/1704.05179.pdf)
- Let us first take a step back, and ask what a full
end-to-end pipeline for question-answering would
look like. A general question-answering system
would be able to answer a question about any domain, based on the world knowledge. This system
would consist of three stages. A given question is
read and reformulated in the first stage, followed
by information retrieval via a search engine. An
answer is then synthesized based on the query and
a set of retrieved documents.
- We notice a gap between the existing closedworld question-answering data sets and this conceptual picture of a general question-answering
system. The general question-answering system
must deal with a noisy set of retrieved documents,
which likely consist of many irrelevant documents as well as semantically and syntactically illformed documents. On the other hand, most of the
existing closed-world question-answering datasets
were constructed in a way that the context provided for each question is guaranteed relevant and
well-written. This guarantee comes from the fact
that each question-answer-context tuple was generated starting from the context from which the
question and answer were extracted.
- we start by
building a set of question-answer pairs from Jeopardy!. We augment each question-answer pair,
which does not have any context attached to it,
by querying Google with the question. This process enables us to retrieve a realistic set of relevant/irrelevant documents, or more specifically
their snippets.

### [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/pdf/1705.03551.pdf)
- Our evidence
documents are automatically gathered from either
Wikipedia or more general Web search results (details in Section 3). Because we gather evidence
using an automated process, the documents are
not guaranteed to contain all facts needed to answer the question. Therefore, they are best seen
as a source of distant supervision, based on the
assumption that the presence of the answer string
in an evidence document implies that the document does answer the question.3 Section 4 shows
that this assumption is valid over 75% of the time,
making evidence documents a strong source of
distant supervision for training machine reading
systems
- Wikipedia pages for entities mentioned in the
question often provide useful information. Wikipedia entities mentioned in the question, and
added the corresponding pages as evidence documents.
- About 73.5%
of these questions contain phrases that describe a
fine grained category to which the answer belongs,
while 15.5% hint at a coarse grained category (one
of person, organization, location, and miscellaneous).
-  Question-evidence pairs in TriviaQA display more lexical and syntactic variance
than SQuAD. This supports our earlier assertion
that decoupling question generation from evidence
collection results in a more challenging problem
- The poor performance of the
random entity baseline shows that the task is not
already solved by information retrieval.
- This suggests that longer compositional questions are harder for current methods
- We randomly
sampled 100 incorrect BiDAF predictions from
the development set and used Wikipedia evidence
documents for manual analysis. We found that 19
examples lacked evidence in any of the provided
documents, 3 had incorrect ground truth, and 3
were valid answers that were not included in the
answer key. Furthermore, 12 predictions were partially correct (Napoleonic vs Napoleonic Wars)
- The first two rows suggest that long and noisy
documents make the question answering task
more difficult, as compared for example to the
short passages in SQuAD
- The crucial difference between
SQuAD/NewsQA and TriviaQA is that TriviaQA
questions have not been crowdsourced from preselected passages
- Knowledge base question answering involves
converting natural language questions to logical
forms that can be executed over a KB. Proposed
datasets (Cai and Yates, 2013; Berant et al., 2013;
Bordes et al., 2015) are either limited in scale or in
the complexity of questions, and can only retrieve
facts covered by the KB.

我们发现Open-domain QA慢慢引起大家重视和兴趣是因为现在传统的问答任务的模型表现越来越好了，所以大家想结合信息检索和传统问答需要的机器阅读理解两个技术来研究open-domain QA，使得问答任务变得更具有普遍性和易使用性，因为此时不需要直接给予阅读理解模型包含所有能回答上问题的推理信息的阅读材料，而需要利用信息检索技术去搜寻和问题相关的材料。然而，信息检索在open-domain QA上的表现并没有很好，这就导致ODQA并没有传统的问答任务的表现好，因为如今和问题相关的阅读材料要么是经由search engine检索出来的文章或者是利用传统的IR技术在大范围材料中进行检索，这些IR模型一个很大的优点就是检索速度非常快（多快？）。但是这些检索出来的文章可能包含能回答上问题的答案和信息，也有可能不包含，所以这样的传统IR模型的检索能力有待考究和加强。而NLP researchers普遍希望通过模型的RC能力来提高ODQA模型的整体表现，这是片面的，因为RC只是ODQA的一部分。随着不同难度以及不同类型的QA dataset的出现，它们对MRC提出的要求也越来越高，因此很多NLP researchers也关注于研究新的reading comprehension model来在这些QA dataset上取得更好的表现。但是同样的研究力度和兴趣并没有给予到IR model上，researchers往往都先从reading comprehension方面来对不同种类不同难度的questions进行突破。所以几乎没有人从IR的角度去研究这些不同难度不同种类的question对IR的影响是什么，从而再根据研究和发现来改善如今open-domain QA里IR部分的能力。

如今，也有很多工作对IR进行了研究，比如增加了不同的neural ranker或者是多重传统IR模型结合或者是利用强化学习和多任务学习等技术加强IR部分，表现都有所上升。但是，这些研究没有先对根本的问题进行研究，也就是这些工作都是在open-domain QA使用的传统IR模型之后做了很多工作加强检索效果，但是传统的IR模型主要在用户提出的哪些种类的问题上表现差，原因又是什么，这些因素目前没有文章去研究。如果能尽量准确、全面和详细的考察不同种类的问题对IR模型的影响，同时挖掘IR模型主要在哪些问题下表现不好的原因，可能会对之后对IR模型的研究和改善有帮助，因为找到了主要导致IR模型表现差的问题类型以及原因，之后在技术的改善上也有了针对性，便可以大幅度的去提升整个open-domain QA里IR部分的表现，使得整体表现更好。