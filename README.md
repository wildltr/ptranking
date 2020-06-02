This open-source project, referred to as **PT-Ranking** (Learning to Rank in PyTorch) aims to provide scalable and extendable implementations of typical learning-to-rank methods based on PyTorch. On one hand, this project enables a uniform comparison over several benchmark datasets leading to an in-depth understanding of previous learning-to-rank methods. On the other hand, this project makes it easy to develop and incorporate newly proposed models, so as to expand the territory of techniques on learning-to-rank. 

# Develop your own learning-to-rank model based on PT-Ranking

PT-Ranking offers deep neural networks as the basis to construct a scoring function based on PyTorch and can thus fully leverage the advantages of PyTorch. 
NeuralRanker is a class that represents a general learning-to-rank model. 
A key component of NeuralRanker is the neural scoring function. The configurable hyper-parameters include activation function, number of layers, number of neurons per layer, etc. 
All specific learning-to-rank models inherit NeuralRanker and mainly differ in the way of computing the training loss.
 The following figure shows the main step in developing a new learning-to-rank model based on Empirical Risk Minimization, 
 where batch_preds and batch_stds correspond to outputs of the scoring function and ground-truth lables, respectively. 
 We can observe that the main work is to define the surrogate loss function.

![NewLoss](./img/new_loss.png)

# Implementations of published learning-to-rank models & References

- Optimization based on Empirical Risk Minimization

| |Model|
|:----|:----|
| Pointwise | RankMSE |
| Pairwise  | RankNet |
| Listwise  | LambdaRank ・ ListNet ・ ListMLE ・ RankCosine ・  ApproxNDCG ・  WassRank ・ STListNet | 
    
- Adversarial Optimization

| |Model|
|:----|:----|
| Pointwise | IR_GAN_Point |
| Pairwise  | IR_GAN_Pair |
| Listwise  | IR_GAN_List |
    
- Tree-based Model (provided by LightGBM & XGBoost)

| |Model|
|:----|:----|
| Listwise | LambdaMART(L)  ・ LambdaMART(X) |

### References

- **RankNet**: Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.

- **RankSVM**: Joachims, Thorsten. Optimizing Search Engines Using Clickthrough Data. Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 133–142, 2002.

- **LambdaRank**: Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006. Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.

- **ListNet**: Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.

- **ListMLE**: Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm. In Proceedings of the 25th ICML. 1192–1199.

- **RankCosine**: Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008. Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.

- **AppoxNDCG**: Tao Qin, Tie-Yan Liu, and Hang Li. 2010. A general approximation framework for direct optimization of information retrieval measures. Journal of Information Retrieval 13, 4 (2010), 375–397.

- **LambdaMART**: Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures. Journal of Information Retrieval, 2007.
(We note that the implementations are provided by [LightGBM](https://lightgbm.readthedocs.io/en/latest/)  and [XGBoost](https://xgboost.readthedocs.io/en/latest/))

- **IRGAN**: Wang, Jun and Yu, Lantao and Zhang, Weinan and Gong, Yu and Xu, Yinghui and Wang, Benyou and Zhang, Peng and Zhang, Dell. IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 515–524, 2017. (**Besides the pointwise and pairiwse adversarial learning-to-rank methods introduced in the paper, we also include the listwise version in PT-Ranking**)

- **WassRank**: Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory. Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 24-32, 2019.

- Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc. A Stochastic Treatment of Learning to Rank Scoring Functions. Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM), 61–69, 2020. 

# Supported Datasets and Formats

## The widely used benchmark datasets listed below can be directly used once downloaded

-- **[LETOR4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)**
(including MQ2007 \| MQ2008 \| MQ2007-semi \| MQ2008-semi \| MQ2007-list \| MQ2008-list )

-- **[MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/)** (including MSLR-WEB10K \| MSLR-WEB30K)

-- **[Yahoo! LETOR](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c)** (including Set1 \| Set2)

-- **[Istella](http://quickrank.isti.cnr.it/istella-dataset/)** (including Istella-S \| Istella \| Istella-X)

### Some notes on the above datasets

- Semi-supervised datasets (MQ2007-semi | MQ2008-semi) have the same format as that for supervised ranking setting. The only difference is that the semi-supervised datasets in this setting contain both judged and undged query-document pairs
 (**in training set but not in validation and testing set**)(The relevance label “-1” indicates the query-document pair is not judged) while the datasets for supervised ranking contain only judged query-document pair.

- According to [Introducing LETOR 4.0 Datasets}](https://arxiv.org/abs/1306.2597), queryLevelNorm version refers to that: conduct query level normalization in the way of using MIN. This data can be directly used for learning. 
They further provide 5-fold partitions of this version for cross fold validation. Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}. 
But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is **not conducted yet**.

- For Yahoo! LETOR, the query-level normalization is already done.

## PT-Ranking currently supports to ingest data with the LibSVM formats

- LETOR datasets in LibSVM format

\<ground-truth label int\> qid\:<query_id int> <feature_id int>:<feature_value float> ... <feature_id int>:<feature_value float>

For example:

4 qid:105 2:0.4  8:0.7   50:0.5


1 qid:105 5:0.5  30:0.7  32:0.4  48:0.53

0 qid:210 4:0.9  38:0.01 39:0.5  45:0.7

1 qid:210 1:0.2  8:0.9   31:0.93 40:0.6

The above sample dataset includes two queries, the query "105" has 2 documents, the corresponding ground-truth labels are 4 and 1, respectively.

- Converting LETOR datasets into LibSVM format with a corresponding **group** file

This functionality is required when using the implementation of LambdaMART provided in [LightGBM](https://lightgbm.readthedocs.io/en/latest/)  and [XGBoost](https://xgboost.readthedocs.io/en/latest/).


# Demo Scripts

- [Jupyter Notebook example with RankNet](./demo/pt_ranking_default_ltr.ipynb)



# Test Setting

PyTorch (1.3)

Python (3.7)

# Call for Contribution
Anyone who are interested in the following kinds of contributions and/or collaborations are warmly welcomed.

**Contribution**: Adding one or more implementations of learning-to-rank models based on the current code base.

# Relevant Resources

| Name | Language | Deep Learning |
|---|---|---|
| [PT-Ranking](https://pt-ranking.github.io/) | Python  |  [PyTorch](https://pytorch.org) | 
| [TF-Ranking](https://github.com/tensorflow/ranking)  |  Python | [TensorFlow](https://tensorflow.org) |
| [MatchZoo](https://github.com/NTMC-Community/MatchZoo) | Python  | [Keras](https://github.com/keras-team/keras) / [PyTorch](https://pytorch.org) | 
| [Shoelace](https://github.com/rjagerman/shoelace)  |  Python | [Chainer](https://chainer.org) |
| [LEROT](https://bitbucket.org/ilps/lerot) | Python  | x  |
| [Rank Lib](http://www.lemurproject.org/ranklib.php) |  Java | x  |
| [Propensity SVM^Rank](http://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html)  |  C | x  |
| [QuickRank](http://quickrank.isti.cnr.it)  |  C++ | x  |
