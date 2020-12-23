## Configuration Strategy

An easy-to-use configuration is necessary for any ML library. PT-Ranking offers a self-contained strategy.
In other words, we appeal to particularly designed class objects for setting. For example, **DataSetting** for data loading, **EvalSetting** for evaluation setting and **ModelParameter** for a model's parameter setting. Moreover, configuration with **json files** is also supported for DataSetting, EvalSetting and ModelParameter, which is the recommended way.

Thanks to this strategy, on one hand, we can initialize the settings for data-loading, evaluation, and models in a simple way. In particular, the parameter setting of a model is self-contained, and easy to customize. On the other hand, given the specified setting, e.g., settings with json files, it is very easy to reproduce an experiment.

## Setting on Loading A Dataset

When loading a dataset, the meta-information and preprocessing are specified with **[DataSetting](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/eval/parameter.py)**. Taking the json file for initializing DataSetting for example,

- "data_id":"MQ2008_Super", # the ID of an adopted dataset
- "dir_data":"/Users/dryuhaitao/WorkBench/Corpus/LETOR4.0/MQ2008/", # the location of an adopted dataset

- "min_docs":[10], # the minimum number of documents per query. Otherwise, the query instance is not used.
- "min_rele":[1], # the minimum number of relevant documents. Otherwise, the query instance is not used.
- "sample_rankings_per_q":[1], # the sample rankings per query

- "binary_rele":[false], # whether convert multi-graded labels into binary labels
- "unknown_as_zero":[false], # whether convert '-1' (i.e., unlabeled documents) as zero
- "presort":[true] # whether sort documents based on ground-truth labels in advance

## Setting on Evaluation

When testing a specific learning-to-rank method, the evaluation details are specified with **[EvalSetting](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/eval/parameter.py)**. Taking the json file for initializing EvalSetting for example,

- "dir_output":"/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/", # output directory of results

- "epochs":100, # the number of epoches for training

- "do_validation":true, # perform validation or not

- "vali_k":5, # the cutoff value for validation, e.g., nDCG@5
- "cutoffs":[1, 3, 5, 10, 20, 50], # the cutoff values for evaluation

- "loss_guided":false, # whether the selection of trained model is based on loss function or validation

- "do_log":true, # logging the training outputs
- "log_step":2, # the step-size of logging
- "do_summary":false,

- "mask_label":false, # mask ground-truth labels
- "mask_type":["rand_mask_all"],
- "mask_ratio":[0.2]

## Parameter Setting

### Parameters for Base Scoring Function
For most learning-to-rank methods, PT-Ranking offers deep neural networks as the basis to construct a scoring function. Therefore, we use [ScoringFunctionParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/eval/parameter.py) to specify the details, such as the number of layers and activation function. Taking the json file for initializing ScoringFunctionParameter for example,

- "BN":[true], # batch normalization
- "RD":[false], # residual module
- "layers":[5], # number of layers
- "apply_tl_af":[true], # use activation function for the last layer
- "hd_hn_tl_af":["R"] # type of activation function

### Parameters for Loss Function
Besides the configuration of the scoring function, for some relatively complex learning-to-rank methods, we also need to specify some parameters for the loss function. In this case, it is required to develop the subclass ModelAParameter by inheriting **[ModelParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/eval/parameter.py)** and customizing the functions, such as to_para_string(), default_para_dict() and grid_search(). Please refer to [LambdaRankParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/listwise/lambdarank.py) as an example.

## Prepare Configuration Json Files

When evaluating a method, two json files are commonly required:

- **Data_Eval_ScoringFunction.json**, which specifies the detailed settings for data loading (i.e, DataSetting), evaluation (i.e, EvalSetting) and the parameters for base scoring function (i.e, ScoringFunctionParameter). Please refer to [Data_Eval_ScoringFunction](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/testing/ltr_adhoc/json/) as an example.

- **XParameter**, which specifies the parameters for model **X**. Please refer to [LambdaRankParameter](https://github.com/ptranking/ptranking.github.io/raw/master/ptranking/ltr_adhoc/listwise/lambdarank.py) as an example.
