## Requirements

1. Prepare a virtual environment with Python 3.*  via `conda`, `venv` or others.

2. Install Pytorch following the [instructions](https://pytorch.org/get-started/locally/)

3. Install scikit-learn following the [instructions](https://scikit-learn.org/stable/install.html#installation-instructions)

## Command-line Usage

1. Download source code
   
```
git clone https://github.com/wildltr/ptranking
```
2. Download [Supported Datasets](../data/data.md), such as **MQ2008** and unrar the .rar file.

```
wget "https://lyoz5a.ch.files.1drv.com/y4mM8g8v4d2mFfO5djKT-ELADpDDRcsVwXRSaZu-9rlOlgvW62Qeuc8hFe_wr6m5NZMnUSEfr6QpMP81ZIQIiwI4BnoHmIZT9Sraf53AmhhIfLi531DOKYZTy4MtDHbBC7dn_Z9DSKvLJZhERPIamAXCrONg7WrFPiG0sTpOXl3-YEYZ1scTslmNyg2a__3YalWRMyEIipY56sy97pb68Sdww" -O MQ2008.rar
```
3. Prepare the required json files (Data_Eval_ScoringFunction.json and XParameter.json, please refer to [Configuration](./Configuration.md) for more information) for specifying evaluation details.
   
4. Run the following command script on your Terminal/Command Prompt:

(1) Without using GPU
```
python pt_ranking.py -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/
```
(2) Using GPU
```
python pt_ranking.py -cuda 0 -model ListMLE -dir_json /home/dl-box/WorkBench/Dropbox/CodeBench/GitPool/wildltr_ptranking/testing/ltr_adhoc/json/
```
The meaning of each supported argument is:
```
optional arguments:
  -h, --help          show this help message and exit
  -cuda CUDA          specify the gpu id if needed, such as 0 or 1.
  -model MODEL        specify the learning-to-rank method
  -debug              quickly check the setting in a debug mode
  -dir_json DIR_JSON  the path of json files specifying the evaluation
                      details.
```

## PyCharm Usage

1. Install [PyCharm](https://www.jetbrains.com/pycharm/) (either Professional version or Community version)
   
2. Download source code
   
```
git clone https://github.com/wildltr/ptranking
```
3. Open the unzipped source code with PyCharm as a new project

4. Test the supported learning-to-rank models by selectively running the following files, where the setting arguments can be changed accordingly
```
testing/ltr_adhoc/testing_ltr_adhoc.py
testing/ltr_adversarial/testing_ltr_adversarial.py
testing/ltr_tree/testing_ltr_tree.py
```

## Python-package Usage

TBA

Install ptranking: pip install ptranking

## Demo Scripts

To get a taste of learning-to-rank models without writing any code, you could try the following script. You just need to specify the model name, the dataset id, as well as the directories for input and output.

- [Jupyter Notebook example on RankNet & LambdaRank](https://github.com/ptranking/ptranking.github.io/raw/master/tutorial/)

To get familiar with the process of data loading, you could try the following script, namely, get the statistics of a dataset.

- [Jupyter Notebook example on getting dataset statistics](https://github.com/ptranking/ptranking.github.io/raw/master/tutorial/)
