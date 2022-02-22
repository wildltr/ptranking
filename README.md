# Introduction

This open-source project, referred to as **PTRanking** (Learning-to-Rank in PyTorch) aims to provide scalable and extendable implementations of typical learning-to-rank methods based on PyTorch. On one hand, this project enables a uniform comparison over several benchmark datasets leading to an in-depth understanding of previous learning-to-rank methods. On the other hand, this project makes it easy to develop and incorporate newly proposed models, so as to expand the territory of techniques on learning-to-rank. 

**Key Features**:

- A number of representative learning-to-rank models for addressing **Ad-hoc Ranking** and **Search Result Diversification**, including not only the traditional optimization framework via empirical risk minimization but also the adversarial optimization framework
- Supports widely used benchmark datasets. Meanwhile, random masking of the ground-truth labels with a specified ratio is also supported
- Supports different metrics, such as Precision, MAP, nDCG, nERR, alpha-nDCG and ERR-IA.
- Highly configurable functionalities for fine-tuning hyper-parameters, e.g., grid-search over hyper-parameters of a specific model
- Provides easy-to-use APIs for developing a new learning-to-rank model

Please refer to the [documentation site](https://wildltr.github.io/ptranking/) for more details.
