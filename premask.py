import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, degree, maybe_num_nodes

class TopologyAdaptiveMask(nn.Module):
    def __init__(self, threshold=0.5, recovery_rate=0.7, total_steps=500, beta=1.0):
        super().__init__()
        self.threshold = threshold
        self.q = recovery_rate
        self.total_steps = total_steps
        self.beta = beta
        self.register_buffer('step', torch.tensor(0))

    def step_update(self):
        self.step += 1

    def compute_similarity(self, x, edge_index):
        row, col = edge_index
        sim = F.cosine_similarity(x[row], x[col], dim=1)
        return sim

    def forward(self, x, edge_index):
        self.step_update()
        sim = self.compute_similarity(x, edge_index)
        mask = sim >= self.threshold
        kept_edges = edge_index[:, mask]
        kept_sim = sim[mask]
        kept_set = set(map(tuple, kept_edges.t().tolist()))
        all_set = set(map(tuple, edge_index.t().tolist()))
        pruned_list = [e for e in all_set if e not in kept_set]
        if len(pruned_list) == 0:
            return kept_edges, kept_edges, torch.empty((2, 0), dtype=torch.long, device=x.device)
        pruned_edges = torch.tensor(pruned_list, device=x.device).t()
        pruned_sim = self.compute_similarity(x, pruned_edges)
        tau = 1 - (self.step / self.total_steps) ** self.beta
        sorted_idx = torch.argsort(pruned_sim, descending=True)
        n_recover = int(self.q * tau * len(pruned_sim))
        recover_idx = sorted_idx[:n_recover]
        recovered = pruned_edges[:, recover_idx]
        A_imp = torch.cat([kept_edges, recovered], dim=1)
        A_imp = coalesce(A_imp, num_nodes=x.size(0))
        A_pur = kept_edges
        discarded_mask = torch.ones(pruned_edges.size(1), dtype=torch.bool, device=x.device)
        discarded_mask[recover_idx] = False
        discarded = pruned_edges[:, discarded_mask]
        return A_pur, A_imp, discarded