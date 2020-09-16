# Learning to Rank in PyTorch

## Introduction

This open-source project, referred to as **PTRanking** (Learning to Rank in PyTorch) aims to provide scalable and extendable implementations of typical learning-to-rank methods based on PyTorch. On one hand, this project enables a uniform comparison over several benchmark datasets leading to an in-depth understanding of previous learning-to-rank methods. On the other hand, this project makes it easy to develop and incorporate newly proposed models, so as to expand the territory of techniques on learning-to-rank.

**Key Features**:

- A number of representative learning-to-rank models, including not only the traditional optimization framework via empirical risk minimization but also the adversarial optimization framework
- Supports widely used benchmark datasets. Meanwhile, random masking of the ground-truth labels with a specified ratio is also supported
- Supports different metrics, such as Precision, MAP, nDCG and nERR
- Highly configurable functionalities for fine-tuning hyper-parameters, e.g., grid-search over hyper-parameters of a specific model
- Provides easy-to-use APIs for developing a new learning-to-rank model

## How-to-Start and Learning more

### Get Started

1. Prepare a virtual environment with Python 3.*  via `conda`, `venv` or others.

2. Install Pytorch following the [instructions](https://pytorch.org/get-started/locally/)

3. Install scikit-learn following the [instructions](https://scikit-learn.org/stable/install.html#installation-instructions)

4. Install ptranking: pip install ptranking

### Command-line Usage

1. Download [Supported Datasets](./data.md)

2. Run the following command script on your Terminal/Command Prompt:

```
python ptranking -data [DATASETNAME] -dir_data [DATASET_DIR] -dir_output [OUTPUT_DIR] -model [MODELNAME]
```

e.g.:
```
python ptranking -data MQ2008_Super -dir_data /home/dl-box/dataset/MQ2008/ -dir_output /home/dl-box/WorkBench/CodeBench/Project_output/Out_L2R/Listwise/ -model ListMLE
```

### Demo Scripts

To get a taste of learning-to-rank models without writing any code, you could try the following script. You just need to specify the model name, the dataset id, as well as the directories for input and output.

- [Jupyter Notebook example on RankNet & LambdaRank](example/ptranking_demo_ltr.ipynb)

To get familiar with the process of data loading, you could try the following script, namely, get the statistics of a dataset.

- [Jupyter Notebook example on getting dataset statistics](example/ptranking_demo_dataset_statistics.ipynb)

### Develop a new model

PT-Ranking offers deep neural networks as the basis to construct a scoring function based on PyTorch and can thus fully leverage the advantages of PyTorch.
NeuralRanker is a class that represents a general learning-to-rank model.
A key component of NeuralRanker is the neural scoring function. The configurable hyper-parameters include activation function, number of layers, number of neurons per layer, etc.
All specific learning-to-rank models inherit NeuralRanker and mainly differ in the way of computing the training loss.
 The following figure shows the main step in developing a new learning-to-rank model based on Empirical Risk Minimization,
 where batch_preds and batch_stds correspond to outputs of the scoring function and ground-truth lables, respectively.
 We can observe that the main work is to define the surrogate loss function.

![](https://github.com/ptranking/ptranking.github.io/raw/master/img/new_loss.png)

### Parameter Setting

An easy-to-use parameter setting is necessary for any ML library. PT-Ranking offers a self-contained strategy.
In other words, we appeals to particularly designed class objects for setting. For example, **DataSetting** for data loading, **EvalSetting** for evaluation setting and **ModelParameter** for a model's parameter setting.

When incorporating a newly developed model (say ModelA), it is commonly required to develop the subclass ModelAParameter by inheriting **[ModelParameter](ptranking/eval/parameter.py)** and customizing the functions, such as to_para_string(), default_para_dict() and grid_search(). Please refer to [LambdaRankParameter](ptranking/ltr_adhoc/listwise/lambdarank.py) as an example.

Thanks to this strategy, on one hand, we can initialize the settings for data-loading, evaluation, and models in a simple way. On the other hand, the parameter setting of a model is self-contained, and easy to customize.

To fully leverage PT-Ranking, one needs to [be familiar with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

For detailed introduction on learning-to-rank, please refer to the book: [Learning to Rank for Information Retrieval](https://link.springer.com/book/10.1007/978-3-642-14267-3).

### Source Code

Please refer to the Github Repository [PT-Ranking](https://github.com/wildltr/ptranking/) for detailed implementations.

## Implemented models

- Optimization based on Empirical Risk Minimization

| |Model|
|:----|:----|
| Pointwise | RankMSE |
| Pairwise  | RankNet |
| Listwise  | ListNet ・ ListMLE ・ RankCosine ・  LambdaRank ・ ApproxNDCG ・  WassRank ・ STListNet ・ LambdaLoss|

- Adversarial Optimization

| |Model|
|:----|:----|
| Pointwise | IR_GAN_Point |
| Pairwise  | IR_GAN_Pair |
| Listwise  | IR_GAN_List |

- Tree-based Model (provided by LightGBM & XGBoost)

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
