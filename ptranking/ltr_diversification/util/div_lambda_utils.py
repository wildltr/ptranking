
import torch


def get_pairwise_comp_probs(batch_preds, std_q_doc_rele_mat, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j, the subtopic labels are aggregated.
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @return:
    '''
    # standard pairwise differences per-subtopic, i.e., S_{ij}
    subtopic_std_diffs = torch.unsqueeze(std_q_doc_rele_mat, dim=2) - torch.unsqueeze(std_q_doc_rele_mat, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    subtopic_std_Sij = torch.clamp(subtopic_std_diffs, min=-1.0, max=1.0)
    subtopic_std_p_ij = 0.5 * (1.0 + subtopic_std_Sij)
    batch_std_p_ij = torch.mean(subtopic_std_p_ij, dim=0, keepdim=True)

    # computing pairwise differences, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    return batch_p_ij, batch_std_p_ij

def get_prob_pairwise_comp_probs(batch_pairsub_mus, batch_pairsub_vars, q_doc_rele_mat):
    '''
    The difference of two normal random variables is another normal random variable.
    pairsub_mu & pairsub_var denote the corresponding mean & variance of the difference of two normal random variables
    p_ij denotes the probability that d_i beats d_j
    @param batch_pairsub_mus:
    @param batch_pairsub_vars:
    @param batch_std_labels:
    @return:
    '''
    subtopic_std_diffs = torch.unsqueeze(q_doc_rele_mat, dim=2) - torch.unsqueeze(q_doc_rele_mat, dim=1)
    subtopic_std_Sij = torch.clamp(subtopic_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
    subtopic_std_p_ij = 0.5 * (1.0 + subtopic_std_Sij)
    batch_std_p_ij = torch.mean(subtopic_std_p_ij, dim=0, keepdim=True)

    batch_p_ij = 1.0 - 0.5 * torch.erfc(batch_pairsub_mus / torch.sqrt(2 * batch_pairsub_vars))

    return batch_p_ij, batch_std_p_ij