
import torch


def get_pairwise_comp_probs(batch_preds, batch_std_labels, sigma=None):
    '''
    Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
    @param batch_preds:
    @param batch_std_labels:
    @param sigma:
    @return:
    '''
    # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
    batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
    batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

    # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
    # ensuring S_{ij} \in {-1, 0, 1}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    return batch_p_ij, batch_std_p_ij