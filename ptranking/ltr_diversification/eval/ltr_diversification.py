
import os
import sys
import json
import yaml
import datetime
import numpy as np

from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.data.data_utils import SPLIT_TYPE
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_adhoc.eval.parameter import ValidationTape
from ptranking.ltr_diversification.util.div_data import DIVDataset, RerankDIVDataset
from ptranking.metric.metric_utils import metric_results_to_string, get_opt_model

from ptranking.ltr_diversification.eval.div_parameter import DivDataSetting, DivEvalSetting, DivScoringFunctionParameter, DivSummaryTape, DivCVTape
from ptranking.ltr_diversification.score_and_sort.daletor import DALETOR, DALETORParameter
from ptranking.ltr_diversification.score_and_sort.div_prob_ranker import DivProbRanker, DivProbRankerParameter

LTR_DIV_MODEL = ['DALETOR', 'DivLambdaRank', 'DivProbRanker', 'DivSoftRank', 'DivTwinRank']

####
# 1> opt as a grid choice; 2> learning rate; 3> self.b for risk-aware ranking;
# DALETOR 1> temperature; 2> opt
# presort as the argument;
####

class DivLTREvaluator(LTREvaluator):
    def __init__(self, frame_id=LTRFRAME_TYPE.Diversification, cuda=None):
        super(DivLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)
        '''
        Since it is time-consuming to generate the ideal diversified ranking dynamically,
        we make it as a global True.
        '''
        self.presort = True

    def determine_files(self, data_splits=None, fold_k=None):
        #dict_splits = {1:[1], 2:[2], 3:[3], 4:[4], 5:[5]}

        fold_ids = [1, 2, 3, 4, 5]
        file_test = data_splits[fold_ids[fold_k-1]]
        file_vali= data_splits[fold_ids[fold_k-5]]
        file_train = data_splits[fold_ids[fold_k-4]] + data_splits[fold_ids[fold_k-3]] + data_splits[fold_ids[fold_k-2]]

        #print("file_test", file_test)
        #print("file_vali", file_vali)
        #print("file_train", file_train)

        return file_train, file_vali, file_test

    def load_data(self, eval_dict=None, data_dict=None, fold_k=None, discriminator=None):
        """
        We note that it is impossible to perform processing over multiple queries,
        since q_doc_rele_mat may differ from query to query.
        @param eval_dict:
        @param data_dict:
        @param fold_k:
        @return:
        """
        file_train, file_vali, file_test = self.determine_files(data_splits=self.data_splits, fold_k=fold_k)

        fold_dir = data_dict['dir_data'] + 'folder' + str(fold_k) + '/'

        if discriminator is not None:
            train_data = \
                RerankDIVDataset(list_as_file=file_train, split_type=SPLIT_TYPE.Train, fold_dir=fold_dir,
                                 data_dict=data_dict, dictQueryRepresentation=self.dictQueryRepresentation,
                                 dictDocumentRepresentation=self.dictDocumentRepresentation,
                                 dictQueryPermutaion=self.dictQueryPermutaion, presort=self.presort,
                                 dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics, buffer=True,
                                 discriminator=discriminator, eval_dict=eval_dict)
            test_data = \
                RerankDIVDataset(list_as_file=file_test, split_type=SPLIT_TYPE.Test, fold_dir=fold_dir,
                                 data_dict=data_dict, dictQueryRepresentation=self.dictQueryRepresentation,
                                 dictQueryPermutaion=self.dictQueryPermutaion,
                                 dictDocumentRepresentation=self.dictDocumentRepresentation,
                                 dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics,
                                 presort=self.presort, discriminator=discriminator, buffer=True, eval_dict=eval_dict)

            vali_data = \
                RerankDIVDataset(list_as_file=file_vali, split_type=SPLIT_TYPE.Validation, fold_dir=fold_dir,
                                 data_dict=data_dict, dictQueryRepresentation=self.dictQueryRepresentation,
                                 dictDocumentRepresentation=self.dictDocumentRepresentation,
                                 dictQueryPermutaion=self.dictQueryPermutaion,
                                 dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics,
                                 buffer=True, presort=self.presort, discriminator=discriminator, eval_dict=eval_dict)
        else:
            train_data = \
                DIVDataset(list_as_file=file_train, split_type=SPLIT_TYPE.Train, fold_dir=fold_dir, data_dict=data_dict,
                           dictQueryRepresentation=self.dictQueryRepresentation,
                           dictDocumentRepresentation=self.dictDocumentRepresentation,
                           dictQueryPermutaion=self.dictQueryPermutaion, buffer=True, presort=self.presort,
                           dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics,
                           add_noise=data_dict['add_noise'], std_delta=data_dict['std_delta'])

            test_data = \
                DIVDataset(list_as_file=file_test, split_type=SPLIT_TYPE.Test, fold_dir=fold_dir,
                           data_dict=data_dict, dictQueryRepresentation=self.dictQueryRepresentation,
                           dictDocumentRepresentation=self.dictDocumentRepresentation,
                           dictQueryPermutaion=self.dictQueryPermutaion, presort=self.presort,
                           dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics, buffer=True,
                           add_noise=data_dict['add_noise'], std_delta=data_dict['std_delta'])

            vali_data = \
                DIVDataset(list_as_file=file_vali, split_type=SPLIT_TYPE.Validation, fold_dir=fold_dir,
                           data_dict=data_dict, dictQueryRepresentation=self.dictQueryRepresentation,
                           dictDocumentRepresentation=self.dictDocumentRepresentation,
                           dictQueryPermutaion=self.dictQueryPermutaion, presort=self.presort,
                           dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics, buffer=True,
                           add_noise=data_dict['add_noise'], std_delta=data_dict['std_delta'])

        return train_data, test_data, vali_data

    def save_as_qrels(self, dictQueryPermutaion, dictQueryDocumentSubtopics, dir=None, data_id=None):

        target_file = '/'.join([dir, data_id+'_qrels.txt'])
        if os.path.isfile(target_file):
            return
        else:
            qrels_writer = open(target_file, 'w')

            for q_id in dictQueryDocumentSubtopics.keys():
                q_doc_subtopics = dictQueryDocumentSubtopics[q_id]
                perm_docs = dictQueryPermutaion[q_id]['permutation']

                # get max subtopic_id
                max_subtopic_id = 0
                for list_subtopic_id in q_doc_subtopics.values():
                    for subtopic_id in list_subtopic_id:
                        if int(subtopic_id) > max_subtopic_id:
                            max_subtopic_id = int(subtopic_id)

                # generate qrels
                for doc in perm_docs:
                    if doc not in q_doc_subtopics:
                        for i in range(1, max_subtopic_id+1):
                            qrels_writer.write(' '.join([q_id, str(i), doc, "0\n"]))
                    else:
                        covered_subtopics = q_doc_subtopics[doc]
                        if len(covered_subtopics) == 0:
                            for i in range(1, max_subtopic_id + 1):
                                qrels_writer.write(' '.join([q_id, str(i), doc, "0\n"]))
                        else:
                            for i in range(1, max_subtopic_id + 1):
                                if str(i) in covered_subtopics:
                                    qrels_writer.write(' '.join([q_id, str(i), doc, "1\n"]))
                                else:
                                    qrels_writer.write(' '.join([q_id, str(i), doc, "0\n"]))

            #==
            qrels_writer.flush()
            qrels_writer.close()


    def load_raw_data(self, eval_dict=None, data_dict=None, fold_k=None):
        root = data_dict['dir_data']
        query_permutation_file = root + 'query_permutation.json'
        query_representation_file = root + 'query_representation.dat'
        document_representation_file = root + 'doc_representation.dat'
        query_document_subtopics_file = root + 'query_doc.json'

        fold_num = 5
        self.data_splits = dict()
        for fold_k in range(1, fold_num + 1):
            with open(root + 'folder'+ str(fold_k) + '/config.yml') as confFile:
                ''' Using the provided splits for a fair comparison '''
                self.data_splits[fold_k] = yaml.load(confFile, Loader=yaml.FullLoader)['test_set']

        #print('self.data_splits', self.data_splits)

        #198
        '''
        total number: 198
        {query_id: {'alphaDCG':*, 'permutation':[list of documents (the number of documents per query is different)]}}
        '''
        with open(query_permutation_file) as self.fileQueryPermutaion:
            self.dictQueryPermutaion = json.load(self.fileQueryPermutaion)
        '''
        num_docs = 0
        for q in self.dictQueryPermutaion.keys():
            #print(self.dictQueryPermutaion[q]['alphaDCG'])
            #print('number of docs: ', len(self.dictQueryPermutaion[q]['permutation']))
            num_docs += len(self.dictQueryPermutaion[q]['permutation'])
        print('num_docs', num_docs)
        '''

        with open(query_representation_file) as self.fileQueryRepresentation:
            self.dictQueryRepresentation = json.load(self.fileQueryRepresentation)
        for query in self.dictQueryRepresentation: # each query is represented as a float vector
            self.dictQueryRepresentation[query] = np.matrix([self.dictQueryRepresentation[query]], dtype=np.float)
            #self.dictQueryRepresentation[query] = np.transpose(self.dictQueryRepresentation[query])

        with open(document_representation_file) as self.fileDocumentRepresentation:
            self.dictDocumentRepresentation = json.load(self.fileDocumentRepresentation)
        for doc in self.dictDocumentRepresentation:
            self.dictDocumentRepresentation[doc] = np.matrix([self.dictDocumentRepresentation[doc]], dtype=np.float)
            #self.dictDocumentRepresentation[doc] = np.transpose(self.dictDocumentRepresentation[doc])

        with open(query_document_subtopics_file) as self.fileQueryDocumentSubtopics:
            self.dictQueryDocumentSubtopics = json.load(self.fileQueryDocumentSubtopics)

        '''
        self.save_as_qrels(dictQueryPermutaion=self.dictQueryPermutaion,
                           dictQueryDocumentSubtopics=self.dictQueryDocumentSubtopics,
                           dir=data_dict['dir_data'], data_id=data_dict['data_id'])
        '''


    def setup_output(self, data_dict=None, eval_dict=None, reproduce=False):
        """
        Update output directory
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        model_id = self.model_parameter.model_id
        grid_search, do_vali, dir_output = eval_dict['grid_search'], eval_dict['do_validation'], eval_dict['dir_output']

        if grid_search or reproduce:
            dir_root = dir_output + '_'.join(['gpu', 'grid', model_id]) + '/' if self.gpu else dir_output + '_'.join(['grid', model_id]) + '/'
        else:
            dir_root = dir_output

        eval_dict['dir_root'] = dir_root
        if not os.path.exists(dir_root): os.makedirs(dir_root)

        sf_str = self.sf_parameter.to_para_string()
        data_eval_str = '_'.join([self.data_setting.to_data_setting_string(),
                                  self.eval_setting.to_eval_setting_string()])

        file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        model_para_string = self.model_parameter.to_para_string()
        if len(model_para_string) > 0:
            dir_run = dir_run + model_para_string + '/'

        eval_dict['dir_run'] = dir_run
        if not os.path.exists(dir_run):
            os.makedirs(dir_run)

        return dir_run

    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        sf_para_dict[sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))

        self.dir_run  = self.setup_output(data_dict, eval_dict)

        if eval_dict['do_log'] and not self.eval_setting.debug:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            sys.stdout = open(self.dir_run + '_'.join(['log', time_str]) + '.txt', "w")

        #if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')


    def div_cv_reproduce(self, data_dict=None, eval_dict=None, sf_para_dict=None, div_para_dict=None):
        self.display_information(data_dict, div_para_dict)

        self.load_raw_data(data_dict=data_dict)
        sf_para_dict[sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))

        model_id = div_para_dict['model_id']
        max_label = data_dict['max_label']
        log_step = eval_dict['log_step']
        vali_metric, vali_k, cutoffs = eval_dict['vali_metric'], eval_dict['vali_k'], eval_dict['cutoffs']
        epochs, do_vali, do_summary = eval_dict['epochs'], eval_dict['do_validation'], eval_dict['do_summary']

        fold_num = 5
        dir_run = self.setup_output(data_dict, eval_dict, reproduce=True)
        cv_tape = DivCVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali,reproduce=True)

        ranker = self.load_ranker(model_para_dict=div_para_dict, sf_para_dict=sf_para_dict)

        for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
            ranker.init()  # initialize or reset with the same random initialization

            _, test_data, _ = self.load_data(data_dict=data_dict, fold_k=fold_k)

            cv_tape.fold_evaluation_reproduce(ranker=ranker, test_data=test_data, dir_run=dir_run,
                                              max_label=max_label, fold_k=fold_k, model_id=model_id)

        andcg_cv_avg_scores = cv_tape.get_cv_performance()  # for comparison among different settings of hyper-parameters
        return andcg_cv_avg_scores

    def load_pretrained_model(self, model, dir_run, fold_k):
        subdir = '-'.join(['Fold', str(fold_k)])
        run_fold_k_dir = os.path.join(dir_run, subdir)
        fold_k_buffered_model_names = os.listdir(run_fold_k_dir)
        fold_opt_model_name = get_opt_model(fold_k_buffered_model_names)
        fold_opt_model = os.path.join(run_fold_k_dir, fold_opt_model_name)
        model.load(file_model=fold_opt_model, context='cpu_gpu')

    def div_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, div_para_dict=None, **kwargs):

        self.display_information(data_dict, div_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, div_para_dict)

        self.load_raw_data(data_dict=data_dict)

        ranker = self.load_ranker(model_para_dict=div_para_dict, sf_para_dict=sf_para_dict)
        ranker.uniform_eval_setting(eval_dict=eval_dict)

        model_id = div_para_dict['model_id']
        max_label = data_dict['max_label']
        log_step = eval_dict['log_step']
        vali_metric, vali_k, cutoffs = eval_dict['vali_metric'], eval_dict['vali_k'], eval_dict['cutoffs']
        epochs, do_vali, do_summary = eval_dict['epochs'], eval_dict['do_validation'], eval_dict['do_summary']

        fold_num = 5
        cv_tape = DivCVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)

        if eval_dict['rerank']:
            d_sf_para_dict, d_div_para_dict = kwargs['d_sf_para_dict'], kwargs['d_div_para_dict']
            d_sf_para_dict[d_sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))
            discriminator = self.load_ranker(model_para_dict=d_div_para_dict, sf_para_dict=d_sf_para_dict)
        else:
            discriminator = None

        for fold_k in range(1, fold_num + 1): # evaluation over k-fold data
            ranker.init() # initialize or reset with the same random initialization

            if eval_dict['rerank']:
                discriminator.init()
                self.load_pretrained_model(model=discriminator, dir_run=eval_dict['rerank_model_dir'], fold_k=fold_k)

            train_data, test_data, vali_data = self.load_data(data_dict=data_dict, fold_k=fold_k, eval_dict=eval_dict,
                                                              discriminator=discriminator)
            if do_vali:
                vali_tape = ValidationTape(num_epochs=epochs, validation_metric=vali_metric, validation_at_k=vali_k,
                                           fold_k=fold_k, dir_run=self.dir_run)
            if do_summary:
                summary_tape = DivSummaryTape(do_validation=do_vali, cutoffs=cutoffs, gpu=self.gpu)

            for epoch_k in range(1, epochs + 1):
                torch_fold_k_epoch_k_loss, stop_training = ranker.div_train(train_data=train_data, epoch_k=epoch_k)
                ranker.scheduler.step()  # adaptive learning rate with step_size=40, gamma=0.5

                if stop_training:
                    print('training is failed !')
                    break

                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:  # per-step validation score
                        vali_metric_value = ranker.div_validation(
                            vali_data=vali_data, vali_metric=vali_metric, k=vali_k, max_label=max_label, device='cpu')
                        vali_tape.epoch_validation(ranker=ranker, epoch_k=epoch_k,
                                                   metric_value=vali_metric_value.squeeze(-1).data.numpy())
                    if do_summary:  # summarize per-step performance w.r.t. train, test
                        summary_tape.epoch_summary(torch_epoch_k_loss=torch_fold_k_epoch_k_loss, ranker=ranker,
                                                   train_data=train_data, vali_data=vali_data, test_data=test_data)

            if do_vali: # loading the fold-wise optimal model for later testing
                ranker.load(vali_tape.get_optimal_path())
                vali_tape.clear_fold_buffer(fold_k=fold_k)
            else: # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

            cv_tape.fold_evaluation(model_id=model_id, ranker=ranker, fold_k=fold_k, test_data=test_data, max_label=max_label)

        andcg_cv_avg_scores = cv_tape.get_cv_performance() # for comparison among different settings of hyper-parameters
        return andcg_cv_avg_scores


    def load_ranker(self, sf_para_dict, model_para_dict):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = model_para_dict['model_id']

        if model_id in ['DALETOR', 'DivLambdaRank', 'DivProbRanker', 'DivSoftRank', 'DivTwinRank']:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                         gpu=self.gpu, device=self.device)
        else:
            raise NotImplementedError

        return ranker

    def log_max(self, data_dict=None, max_cv_avg_scores=None, sf_para_dict=None,  eval_dict=None, log_para_str=None):
        ''' Log the best performance across grid search and the corresponding setting '''
        dir_root, cutoffs = eval_dict['dir_root'], eval_dict['cutoffs']
        data_id = data_dict['data_id']

        sf_str = self.sf_parameter.to_para_string(log=True)

        data_eval_str = self.data_setting.to_data_setting_string(log=True) +'\n'+ self.eval_setting.to_eval_setting_string(log=True)

        with open(file=dir_root + '/' + '_'.join([data_id, sf_para_dict['sf_id'], 'max.txt']), mode='w') as max_writer:
            max_writer.write('\n\n'.join([data_eval_str, sf_str, log_para_str, metric_results_to_string(max_cv_avg_scores, cutoffs, metric='aNDCG')]))

    def set_data_setting(self, debug=False, data_id=None, dir_data=None, div_data_json=None):
        if div_data_json is not None:
            self.data_setting = DivDataSetting(div_data_json=div_data_json)
        else:
            self.data_setting = DivDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, debug=False, dir_output=None, div_eval_json=None):
        if div_eval_json is not None:
            self.eval_setting = DivEvalSetting(debug=debug, div_eval_json=div_eval_json)
        else:
            self.eval_setting = DivEvalSetting(debug=debug, dir_output=dir_output)

    def set_scoring_function_setting(self, debug=None, sf_id=None, sf_json=None):
        if sf_json is not None:
            self.sf_parameter = DivScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = DivScoringFunctionParameter(debug=debug, sf_id=sf_id)

    def set_model_setting(self, debug=False, model_id=None, para_json=None):
        if para_json is not None:
            self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
        else:
            self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def run(self, debug=False, model_id=None, sf_id=None, config_with_json=None,
            dir_json=None, data_id=None, dir_data=None, dir_output=None, grid_search=False, reproduce=False):
        if config_with_json:
            assert dir_json is not None
            if reproduce:
                self.point_run(debug=debug, model_id=model_id, dir_json=dir_json, reproduce=reproduce)
            else:
                self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            assert sf_id in ['pointsf', 'listsf', 'listsf_co']
            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, sf_id=sf_id,
                              data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, sf_id=sf_id,
                               data_id=data_id, dir_data=dir_data, dir_output=dir_output)

    def grid_run(self, debug=True, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None, dir_json=None):
        """
        Perform diversified ranking based on grid search of optimal parameter setting
        """
        if dir_json is not None:
            div_data_eval_sf_json = dir_json + 'Div_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "Parameter.json"
            self.set_eval_setting(debug=debug, div_eval_json=div_data_eval_sf_json)
            self.set_data_setting(div_data_json=div_data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=div_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
            self.set_model_setting(debug=debug, model_id=model_id)

        ''' select the best setting through grid search '''
        vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50] # cutoffs should be consistent w.r.t. eval_dict
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_div_para_dict = None, None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                if eval_dict['rerank']:
                    d_sf_para_dict, d_div_para_dict = self.get_rerank_para_dicts(eval_dict=eval_dict)
                else:
                    d_sf_para_dict, d_div_para_dict = None, None

                for sf_para_dict in self.iterate_scoring_function_setting():
                    for div_para_dict in self.iterate_model_setting():
                        curr_cv_avg_scores = \
                            self.div_cv_eval(data_dict=data_dict, eval_dict=eval_dict,
                                             sf_para_dict=sf_para_dict, div_para_dict=div_para_dict,
                                             d_sf_para_dict=d_sf_para_dict, d_div_para_dict=d_div_para_dict)

                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_div_para_dict = \
                                curr_cv_avg_scores, sf_para_dict, eval_dict, div_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, eval_dict=max_eval_dict,
                     max_cv_avg_scores=max_cv_avg_scores, sf_para_dict=max_sf_para_dict,
                     log_para_str=self.model_parameter.to_para_string(log=True, given_para_dict=max_div_para_dict))


    def get_rerank_para_dicts(self, eval_dict):
        rerank_dir = eval_dict['rerank_dir']
        rerank_model_id = eval_dict['rerank_model_id']
        rerank_div_data_eval_sf_json = rerank_dir + 'Div_Data_Eval_ScoringFunction.json'
        rerank_para_json = rerank_dir + rerank_model_id + "Parameter.json"

        rerank_sf_parameter = DivScoringFunctionParameter(sf_json=rerank_div_data_eval_sf_json)
        rerank_model_parameter = globals()[rerank_model_id + "Parameter"](para_json=rerank_para_json)
        d_sf_para_dict = rerank_sf_parameter.default_para_dict()
        d_div_para_dict = rerank_model_parameter.default_para_dict()

        return d_sf_para_dict, d_div_para_dict


    def point_run(self, debug=False, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None,
                  dir_json=None, reproduce=False):
        """
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """

        if dir_json is None:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
            self.set_model_setting(debug=debug, model_id=model_id)
        else:
            div_data_eval_sf_json = dir_json + 'Div_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "Parameter.json"
            self.set_eval_setting(debug=debug, div_eval_json=div_data_eval_sf_json)
            self.set_data_setting(div_data_json=div_data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=div_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)

        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()
        sf_para_dict = self.get_default_scoring_function_setting()
        div_model_para_dict = self.get_default_model_setting()

        if eval_dict['rerank']:
            d_sf_para_dict, d_div_para_dict = self.get_rerank_para_dicts(eval_dict=eval_dict)
            self.div_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                             div_para_dict=div_model_para_dict,
                             d_sf_para_dict=d_sf_para_dict, d_div_para_dict=d_div_para_dict)
        else:
            if reproduce:
                self.div_cv_reproduce(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                                      div_para_dict=div_model_para_dict)
            else:
                self.div_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                                 div_para_dict=div_model_para_dict)

