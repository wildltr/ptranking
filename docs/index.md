# Learning-to-Rank in PyTorch

## Introduction

This open-source project, referred to as **PTRanking** (Learning-to-Rank in PyTorch) aims to provide scalable and extendable implementations of typical learning-to-rank methods based on PyTorch. On one hand, this project enables a uniform comparison over several benchmark datasets, leading to an in-depth understanding of previous learning-to-rank methods. On the other hand, this project makes it easy to develop and incorporate newly proposed models, so as to expand the territory of techniques on learning-to-rank.

**Key Features**:

- A number of representative learning-to-rank models for addressing **Ad-hoc Ranking** and **Search Result Diversification**, including not only the traditional optimization framework via empirical risk minimization but also the adversarial optimization framework
- Supports widely used benchmark datasets. Meanwhile, random masking of the ground-truth labels with a specified ratio is also supported
- Supports different metrics, such as Precision, MAP, nDCG, nERR, alpha-nDCG and ERR-IA.
- Highly configurable functionalities for fine-tuning hyper-parameters, e.g., grid-search over hyper-parameters of a specific model
- Provides easy-to-use APIs for developing a new learning-to-rank model

### Source Code

Please refer to the Github Repository [PT-Ranking](https://github.com/wildltr/ptranking/) for detailed implementations.

## Implemented models

- Typical Learning-to-Rank Methods for Ad-hoc Ranking

| |Model|
|:----|:----|
| Pointwise | RankMSE |
| Pairwise  | RankNet |
| Listwise  | ListNet ・ ListMLE ・ RankCosine ・  LambdaRank ・ ApproxNDCG ・  WassRank ・ STListNet ・ LambdaLoss|

- Learning-to-Rank Methods for Search Result Diversification

| |Model|
|:----|:----|
| Score-and-sort strategy | MO4SRD ・ DALETOR|

- Adversarial Learning-to-Rank Methods for Ad-hoc Ranking

| |Model|
|:----|:----|
| Pointwise | IR_GAN_Point |
| Pairwise  | IR_GAN_Pair |
| Listwise  | IR_GAN_List |

- Learning-to-rank Methods Based on Gradient Boosting Decision Trees (GBDT) (based on LightGBM)

| |Model|
|:----|:----|
| Listwise | LightGBMLambdaMART |

#### References

- **RankNet**: Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. Learning to rank using gradient descent. In Proceedings of the 22nd ICML. 89–96.

- **RankSVM**: Joachims, Thorsten. Optimizing Search Engines Using Clickthrough Data. Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 133–142, 2002.

- **LambdaRank**: Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006. Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.

- **ListNet**: Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to Rank: From Pairwise Approach to Listwise Approach. In Proceedings of the 24th ICML. 129–136.

- **ListMLE**: Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. 2008. Listwise Approach to Learning to Rank: Theory and Algorithm. In Proceedings of the 25th ICML. 1192–1199.

- **RankCosine**: Tao Qin, Xu-Dong Zhang, Ming-Feng Tsai, De-Sheng Wang, Tie-Yan Liu, and Hang Li. 2008. Query-level loss functions for information retrieval. Information Processing and Management 44, 2 (2008), 838–855.

- **AppoxNDCG**: Tao Qin, Tie-Yan Liu, and Hang Li. 2010. A general approximation framework for direct optimization of information retrieval measures. Journal of Information Retrieval 13, 4 (2010), 375–397.

- **LambdaMART**: Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures. Journal of Information Retrieval, 2007.
(We note that the implementation is provided by [LightGBM](https://lightgbm.readthedocs.io/en/latest/))

- **IRGAN**: Wang, Jun and Yu, Lantao and Zhang, Weinan and Gong, Yu and Xu, Yinghui and Wang, Benyou and Zhang, Peng and Zhang, Dell. IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 515–524, 2017. (**Besides the pointwise and pairiwse adversarial learning-to-rank methods introduced in the paper, we also include the listwise version in PT-Ranking**)

- **LambdaLoss** Xuanhui Wang, Cheng Li, Nadav Golbandi, Mike Bendersky and Marc Najork. The LambdaLoss Framework for Ranking Metric Optimization. Proceedings of The 27th ACM International Conference on Information and Knowledge Management (CIKM '18), 1313-1322, 2018.

- **WassRank**: Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon Jose, Xiao Yang and Long Chen. WassRank: Listwise Document Ranking Using Optimal Transport Theory. Proceedings of the 12th International Conference on Web Search and Data Mining (WSDM), 24-32, 2019.

- Bruch, Sebastian and Han, Shuguang and Bendersky, Michael and Najork, Marc. A Stochastic Treatment of Learning to Rank Scoring Functions. Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM), 61–69, 2020.

- **MO4SRD**: Hai-Tao Yu. Optimize What You EvaluateWith: Search Result Diversification Based on Metric
Optimization. The 36th AAAI Conference on Artificial Intelligence, 2022.

- **DALETOR**: Le Yan, Zhen Qin, Rama Kumar Pasumarthi, Xuanhui Wang, Michael Bendersky. Diversification-Aware Learning to Rank
using Distributed Representation. In Proceedings of the Web Conference 2021, 127–136.

## Test Setting

PyTorch (>=1.3)

Python (3)

Ubuntu 16.04 LTS

## Call for Contribution

We are adding more learning-to-rank models all the time. Please submit an issue if there is something you want to have implemented and included. Meanwhile,
anyone who are interested in any kinds of contributions and/or collaborations are warmly welcomed.

## Citation

If you use PTRanking in your research, please use the following BibTex entry.

```
@misc{yu2020ptranking,
    title={PT-Ranking: A Benchmarking Platform for Neural Learning-to-Rank},
    author={Hai-Tao Yu},
    year={2020},
    eprint={2008.13368},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```

## Relevant Resources

| Name | Language | Deep Learning |
|---|---|---|
| [PTRanking](https://ptranking.github.io/) | Python  |  [PyTorch](https://pytorch.org) |
| [TF-Ranking](https://github.com/tensorflow/ranking)  |  Python | [TensorFlow](https://tensorflow.org) |
| [MatchZoo](https://github.com/NTMC-Community/MatchZoo) | Python  | [Keras](https://github.com/keras-team/keras) / [PyTorch](https://pytorch.org) |
| [Shoelace](https://github.com/rjagerman/shoelace)  |  Python | [Chainer](https://chainer.org) |
| [LEROT](https://bitbucket.org/ilps/lerot) | Python  | x  |
| [Rank Lib](http://www.lemurproject.org/ranklib.php) |  Java | x  |
| [Propensity SVM^Rank](http://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html)  |  C | x  |
| [QuickRank](https://github.com/hpclab/quickrank)  |  C++ | x  |
