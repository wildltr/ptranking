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

### Demo Scripts

To get a taste of learning-to-rank models without writing any code, you could try the following script. You just need to specify the model name, the dataset id, as well as the directories for input and output.

- [Jupyter Notebook example on RankNet & LambdaRank](../example/ptranking_demo_ltr.ipynb)

To get familiar with the process of data loading, you could try the following script, namely, get the statistics of a dataset.

- [Jupyter Notebook example on getting dataset statistics](../example/ptranking_demo_dataset_statistics.ipynb)

### Develop A New Model

PT-Ranking offers deep neural networks as the basis to construct a scoring function based on PyTorch and can thus fully leverage the advantages of PyTorch.
NeuralRanker is a class that represents a general learning-to-rank model.
A key component of NeuralRanker is the neural scoring function. The configurable hyper-parameters include activation function, number of layers, number of neurons per layer, etc.
All specific learning-to-rank models inherit NeuralRanker and mainly differ in the way of computing the training loss.
 The following figure shows the main step in developing a new learning-to-rank model based on Empirical Risk Minimization,
 where batch_preds and batch_stds correspond to outputs of the scoring function and ground-truth lables, respectively.
 We can observe that the main work is to define the surrogate loss function.

![NewLoss](./img/new_loss.png)

### Parameter Setting

An easy-to-use parameter setting is necessary for any ML library. PT-Ranking offers a self-contained strategy.
In other words, we appeals to particularly designed class objects for setting. For example, **DataSetting** for data loading, **EvalSetting** for evaluation setting and **ModelParameter** for a model's parameter setting.

When incorporating a newly developed model (say ModelA), it is commonly required to develop the subclass ModelAParameter by inheriting **[ModelParameter](ptranking/eval/parameter.py)** and customizing the functions, such as to_para_string(), default_para_dict() and grid_search(). Please refer to [LambdaRankParameter](ptranking/ltr_adhoc/listwise/lambdarank.py) as an example.

Thanks to this strategy, on one hand, we can initialize the settings for data-loading, evaluation, and models in a simple way. On the other hand, the parameter setting of a model is self-contained, and easy to customize.

To fully leverage PT-Ranking, one needs to [be familiar with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

For detailed introduction on learning-to-rank, please refer to the book: [Learning to Rank for Information Retrieval](https://link.springer.com/book/10.1007/978-3-642-14267-3).

### Source Codes

Please refer to the [GitHub Repository](https://github.com/ptranking/ptranking.github.io) for PTRanking's implementation details.

## Call for Contribution

We are adding more learning-to-rank models all the time. Please submit an issue if there is something you want to have implemented and included. Meanwhile,
anyone who are interested in any kinds of contributions and/or collaborations are warmly welcomed.

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
