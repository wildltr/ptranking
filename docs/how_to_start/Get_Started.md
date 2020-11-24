## Requirements

1. Prepare a virtual environment with Python 3.*  via `conda`, `venv` or others.

2. Install Pytorch following the [instructions](https://pytorch.org/get-started/locally/)

3. Install scikit-learn following the [instructions](https://scikit-learn.org/stable/install.html#installation-instructions)

4. Install ptranking: pip install ptranking

## Command-line Usage

1. Download [Supported Datasets](./data.md)

2. Run the following command script on your Terminal/Command Prompt:

```
python ptranking -data [DATASETNAME] -dir_data [DATASET_DIR] -dir_output [OUTPUT_DIR] -model [MODELNAME]
```

e.g.:
```
python ptranking -data MQ2008_Super -dir_data /home/dl-box/dataset/MQ2008/ -dir_output /home/dl-box/WorkBench/CodeBench/Project_output/Out_L2R/Listwise/ -model ListMLE
```

## Demo Scripts

To get a taste of learning-to-rank models without writing any code, you could try the following script. You just need to specify the model name, the dataset id, as well as the directories for input and output.

- [Jupyter Notebook example on RankNet & LambdaRank](https://github.com/ptranking/ptranking.github.io/raw/master/tutorial/)

To get familiar with the process of data loading, you could try the following script, namely, get the statistics of a dataset.

- [Jupyter Notebook example on getting dataset statistics](https://github.com/ptranking/ptranking.github.io/raw/master/tutorial/)
