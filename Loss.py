import torch
import torch.nn.functional as F

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def mse_loss(x_hat, x):
    return F.mse_loss(x_hat, x)

def weighted_ce_loss(pred_pos, pred_neg, w_pos, w_neg):
    pos_loss = - w_pos * torch.log(pred_pos + 1e-15).mean()
    neg_loss = - w_neg * torch.log(1 - pred_neg + 1e-15).mean()
    return pos_loss + neg_loss

def subgraph_constraint_loss(h1, h2, tau=1.0):
    h1_norm = F.normalize(h1, p=2, dim=-1)
    h2_norm = F.normalize(h2, p=2, dim=-1)
    cos_sim = (h1_norm * h2_norm).sum(dim=-1)
    loss = (1 - cos_sim).pow(tau).mean()
    return loss

def filtering_loss(z1, z2, mask_vertices):
    if mask_vertices.numel() == 0:
        return torch.tensor(0.0, device=z1.device)
    return F.mse_loss(z1[mask_vertices], z2[mask_vertices])

def clustering_loss(z, cluster_centers, confidence_threshold=0.5):
    dist = torch.cdist(z, cluster_centers, p=2) ** 2
    p = (1 + dist) ** (-1)
    p = p / p.sum(dim=1, keepdim=True)
    p_max, p_idx = p.max(dim=1)
    q = torch.zeros_like(p)
    mask = p_max >= confidence_threshold
    q[mask] = F.one_hot(p_idx[mask], num_classes=cluster_centers.size(0)).float()
    return F.kl_div(p.log(), q, reduction='batchmean')

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), dim=1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()