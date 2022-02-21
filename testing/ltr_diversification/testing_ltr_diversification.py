"""Description
A simple script for testing either in-built methods or newly added methods
"""

import torch
import numpy as np

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_diversification.eval.ltr_diversification import DivLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

if __name__ == '__main__':
    """
    >>> Models <<<
    (1) Learning-to-Rank & Search Result Diversification

    >>> Supported Datasets <<<
    -----------------------------------------------------------------------------------------
    | WebTrack_Div_2009_2012                                                                |
    -----------------------------------------------------------------------------------------

    """

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = True  # in a debug mode, we just check whether the model can operate

    config_with_json = False  # specify configuration with json files or not

    reproduce = False

    models_to_run = [
        #'DALETOR',
        'DivProbRanker',
    ]

    evaluator = DivLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        #dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/drl_ptranking/testing/ltr_diversification/json/'

        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/GE_S/Opt_aNDCG/'
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/GE_S/Opt_nERRIA/'

        # DivProbRanker - SuperSoft - Full
        # dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/GE_Full/Opt_aNDCG/'
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/R_Full/Opt_aNDCG/'

        # reproduce
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/R_reproduce/aNDCG/'
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/R_reproduce/nERRIA/'
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/DivSoftRank_reproduce/aNDCG/'
        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/DivSoftRank_reproduce/nERRIA/'

        #dir_json = '/home/user/T2_Workbench/ExperimentBench/SRD_AAAI/DivLTR/DALETOR_reproduce/'
        #dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/Div_reproduce/aNDCG/' # nERR-IA
        dir_json = '/T2Root/dl-box/T2_WorkBench/ExperimentBench/TwinRank_WWW/Div_reproduce/nERR-IA/'  #


        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json,
                          reproduce=reproduce)

    else:  # specify configuration manually
        sf_id = 'listsf'  # pointsf | listsf | listsf_co, namely the type of neural scoring function

        ''' Selected dataset '''
        data_id = 'WT_Div_0912_Implicit'

        ''' By grid_search, we can explore the effects of different hyper-parameters of a model '''
        grid_search = False

        ''' Location of the adopted data '''
        dir_data = '/Users/iimac/Workbench/Corpus/L2R/TREC_WebTrack_Div_2009_2012_Implicit/'
        #dir_data = '/home/user/T2_Workbench/Corpus/L2R/TREC_WebTrack_Div_2009_2012_Implicit/'

        ''' Output directory '''
        #dir_output = '/home/user/T2_Workbench/Project_output/Out_L2R/DivLTR/'
        dir_output = '/Users/iimac/Workbench/CodeBench/Output/DivLTR/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, sf_id=sf_id, grid_search=grid_search,
                          data_id=data_id, dir_data=dir_data, dir_output=dir_output)
