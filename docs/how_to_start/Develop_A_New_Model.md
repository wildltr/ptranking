## Develop Your Own Learning-to-Rank Method

PT-Ranking offers deep neural networks as the basis to construct a scoring function based on PyTorch and can thus fully leverage the advantages of PyTorch.
NeuralRanker is a class that represents a general learning-to-rank model.
A key component of NeuralRanker is the neural scoring function. The configurable hyper-parameters include activation function, number of layers, number of neurons per layer, etc.
All specific learning-to-rank models inherit NeuralRanker and mainly differ in the way of computing the training loss.
 The following figure shows the main step in developing a new learning-to-rank model based on Empirical Risk Minimization,
 where batch_preds and batch_stds correspond to outputs of the scoring function and ground-truth lables, respectively.
 We can observe that the main work is to define the surrogate loss function.

![](https://github.com/ptranking/ptranking.github.io/raw/master/img/new_loss.png)

When incorporating a newly developed model (say ModelA), it is commonly required to develop the subclass ModelAParameter by inheriting **[ModelParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/eval/parameter.py)** and customizing the functions, such as to_para_string(), default_para_dict() and grid_search(). Please refer to [Configuration](./Configuration.md) for detailed description on parameter setting and [LambdaRankParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/listwise/lambdarank.py) as an example.

To fully leverage PT-Ranking, one needs to [be familiar with PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

For detailed introduction on learning-to-rank, please refer to the book: [Learning to Rank for Information Retrieval](https://link.springer.com/book/10.1007/978-3-642-14267-3).
