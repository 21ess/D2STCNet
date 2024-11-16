import copy
import numpy as np
import torch


def sim_global(flow_data, sim_type='cos'):
    """
    """
    if len(flow_data.shape) == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', flow_data, flow_data)
    elif len(flow_data.shape) == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1  # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', flow_data, flow_data)
    else:
        raise ValueError('sim_global only support shape length in [3, 4] but got {}.'.format(len(flow_data.shape)))

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')

    return sim

def aug_traffic(attn_mx, flow_data, percent=0.2):
    """Generate the data augumentation from traffic (node attribute) perspective.
    :param t_sim_mx: temporal similarity matrix after softmax, [l,n,v]
    :param flow_data: input flow data, [n,l,v,c]
    """
    attn_mx = attn_mx.cpu().detach()
    l, n, v = attn_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = copy.deepcopy(flow_data)

    mask_prob = (1. - attn_mx.permute(1, 0, 2).reshape(-1)).numpy()
    mask_prob = np.nan_to_num(mask_prob, nan=0.0)
    if mask_prob.sum() == 0:
        # avoid 0
        mask_prob = np.ones_like(mask_prob) / len(mask_prob)
    else:
        #
        mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)

    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list],
        y.reshape(-1)[mask_list],
        z.reshape(-1)[mask_list]] = zeros

    return aug_flow