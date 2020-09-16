# Supported Datasets and Formats

## Popular Benchmark Datasets

-- **[LETOR4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)**
(MQ2007 \| MQ2008 \| MQ2007-semi \| MQ2008-semi \| MQ2007-list \| MQ2008-list )

-- **[MSLR-WEB](https://www.microsoft.com/en-us/research/project/mslr/)** (MSLR-WEB10K \| MSLR-WEB30K)

-- **[Yahoo! LETOR](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c)** (including Set1 \| Set2)

-- **[Istella](http://quickrank.isti.cnr.it/istella-dataset/)** (Istella-S \| Istella \| Istella-X)

These above datasets can be directly used once downloaded. **Please note that:**

- Semi-supervised datasets (MQ2007-semi | MQ2008-semi) have the same format as that for supervised ranking setting. The only difference is that the semi-supervised datasets in this setting contain both judged and undged query-document pairs
 (**in training set but not in validation and testing set**)(The relevance label “-1” indicates the query-document pair is not judged) while the datasets for supervised ranking contain only judged query-document pair.

- According to [Introducing LETOR 4.0 Datasets](https://arxiv.org/abs/1306.2597), queryLevelNorm version refers to that: conduct query level normalization in the way of using MIN. This data can be directly used for learning.
They further provide 5-fold partitions of this version for cross fold validation. Thus there is no need to perform query_level_scale again for {MQ2007_super | MQ2008_super | MQ2007_semi | MQ2008_semi}.
But for {MSLRWEB10K | MSLRWEB30K}, the query-level normalization is **not conducted yet**.

- For Yahoo! LETOR, the query-level normalization is already done.

- For Istella! LETOR, the query-level normalization is **not conducted yet**. We note that ISTELLA contains extremely large features, e.g., 1.79769313486e+308, we replace features of this kind with a constant 1000000.

## LibSVM formats

PT-Ranking currently supports to ingest data with the LibSVM formats

- LETOR datasets in LibSVM format

\<ground-truth label int\> qid\:<query_id int> <feature_id int>:<feature_value float> ... <feature_id int>:<feature_value float>

For example:

4 qid:105 2:0.4  8:0.7   50:0.5


1 qid:105 5:0.5  30:0.7  32:0.4  48:0.53

0 qid:210 4:0.9  38:0.01 39:0.5  45:0.7

1 qid:210 1:0.2  8:0.9   31:0.93 40:0.6

The above sample dataset includes two queries, the query "105" has 2 documents, the corresponding ground-truth labels are 4 and 1, respectively.

- Converting LETOR datasets into LibSVM format with a corresponding **group** file

This functionality is required when using the implementation of LambdaMART provided in [LightGBM](https://lightgbm.readthedocs.io/en/latest/).
