import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.utils import coalesce, degree
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class TopologyAdaptiveFilter:
    def __init__(self, threshold=0.5, recovery_rate=0.7, total_steps=500, beta=1.0):
        self.threshold = threshold
        self.q = recovery_rate
        self.total_steps = total_steps
        self.beta = beta
        self.step = 0

    def step_update(self):
        self.step += 1

    def compute_similarity(self, x, edge_index):
        row, col = edge_index
        sim = F.cosine_similarity(x[row], x[col], dim=1)
        return sim

    def prune(self, x, edge_index):
        sim = self.compute_similarity(x, edge_index)
        mask = sim >= self.threshold
        return edge_index[:, mask], sim

    def recover(self, edge_index, sim, pruned_edges, pruned_sim):
        tau = 1 - (self.step / self.total_steps) ** self.beta
        sorted_idx = torch.argsort(pruned_sim, descending=True)
        n_recover = int(self.q * tau * len(pruned_sim))
        recover_idx = sorted_idx[:n_recover]
        recovered = pruned_edges[:, recover_idx]
        final_edges = torch.cat([edge_index, recovered], dim=1)
        final_edges = coalesce(final_edges, num_nodes=x.size(0))
        return final_edges

    def forward(self, x, edge_index):
        kept_edges, kept_sim = self.prune(x, edge_index)
        kept_set = set(map(tuple, kept_edges.t().tolist()))
        all_set = set(map(tuple, edge_index.t().tolist()))
        pruned_list = [e for e in all_set if e not in kept_set]
        if len(pruned_list) == 0:
            return kept_edges
        pruned_edges = torch.tensor(pruned_list, device=x.device).t()
        pruned_sim = self.compute_similarity(x, pruned_edges)
        final_edges = self.recover(kept_edges, kept_sim, pruned_edges, pruned_sim)
        return final_edges

class BoltzmannGibbsMask(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, edge_index):
        N = x.size(0)
        m = torch.randn(N, N, device=x.device)
        M = F.softmax(m / self.alpha, dim=1)
        x_ban = M @ x
        return x_ban

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

class FeatureDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, z):
        return self.mlp(z)

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def reset_parameters(self):
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, z, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([z[src], z[dst]], dim=-1)
        return torch.sigmoid(self.mlp(edge_feat)).squeeze(-1)

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

class DPVGAE(nn.Module):
    def __init__(self, num_features, num_nodes, args):
        super().__init__()
        self.num_nodes = num_nodes
        self.args = args
        self.topo_filter = TopologyAdaptiveFilter(
            threshold=args.topo_threshold,
            recovery_rate=args.recovery_rate,
            total_steps=args.epochs,
            beta=args.beta
        )
        self.feat_mask = BoltzmannGibbsMask(alpha=args.alpha_temp)

        self.encoder_q = GNNEncoder(num_features, args.encoder_channels, args.hidden_channels,
                                     num_layers=args.encoder_layers, dropout=args.encoder_dropout)
        self.encoder_k = GNNEncoder(num_features, args.encoder_channels, args.hidden_channels,
                                     num_layers=args.encoder_layers, dropout=args.encoder_dropout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.inner_product_decoder = InnerProductDecoder()
        self.feature_decoder = FeatureDecoder(args.hidden_channels, num_features, args.decoder_channels)
        self.mask_predictor = MaskPredictor(args.hidden_channels, args.decoder_channels)

        self.gin = GINEncoder(args.hidden_channels, args.decoder_channels, args.decoder_channels, num_layers=2)

        self.num_clusters = args.num_clusters
        self.cluster_centers = nn.Parameter(torch.randn(args.num_clusters, args.hidden_channels))

        self.momentum = args.momentum

    def reset_parameters(self):
        self.encoder_q.reset_parameters()
        self.encoder_k.reset_parameters()
        self.feature_decoder.reset_parameters()
        self.mask_predictor.reset_parameters()
        self.gin.reset_parameters()
        nn.init.xavier_uniform_(self.cluster_centers)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, data):
        x, edge_index_orig = data.x, data.edge_index
        device = x.device

        self.topo_filter.step_update()
        edge_index_pur = self.topo_filter.forward(x, edge_index_orig)
        edge_index_imp = edge_index_orig

        x_ban = self.feat_mask(x, edge_index_orig)

        z_q = self.encoder_q(x_ban, edge_index_imp)
        with torch.no_grad():
            z_k = self.encoder_k(x_ban, edge_index_pur)

        perm = torch.randperm(edge_index_pur.size(1))
        keep = int(0.7 * edge_index_pur.size(1))
        edge_view2 = edge_index_pur[:, perm[:keep]]
        h1 = self.gin(z_k, edge_index_pur)
        h2 = self.gin(z_k, edge_view2)
        h1_norm = F.normalize(h1, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)
        l_fg = - (h1_norm * h2_norm).sum(dim=-1).mean()

        deg_pur = degree(edge_index_pur[0], num_nodes=self.num_nodes)
        mask_vertices = (deg_pur == 0).nonzero().squeeze()
        if mask_vertices.numel() == 0:
            l_pur = torch.tensor(0.0, device=device)
        else:
            l_pur = F.mse_loss(z_q[mask_vertices], z_k[mask_vertices])

        z = z_q
        dist = torch.cdist(z, self.cluster_centers, p=2) ** 2
        p = (1 + dist) ** (-1)
        p = p / p.sum(dim=1, keepdim=True)
        target = p.argmax(dim=1)
        l_cluster = F.kl_div(p.log(), F.one_hot(target, num_classes=self.num_clusters).float(), reduction='batchmean')

        logits = self.inner_product_decoder(z_q, edge_index_orig)
        pos_loss = -F.logsigmoid(logits).mean()
        neg_edge_index = torch.randint(0, self.num_nodes, (2, edge_index_orig.size(1)), device=device)
        neg_logits = self.inner_product_decoder(z_q, neg_edge_index)
        neg_loss = -F.logsigmoid(-neg_logits).mean()
        l1 = pos_loss + neg_loss

        x_hat = self.feature_decoder(z_q)
        l2 = F.mse_loss(x_hat, x)

        kept_set = set(map(tuple, edge_index_pur.t().tolist()))
        all_set = set(map(tuple, edge_index_orig.t().tolist()))
        masked_list = [e for e in all_set if e not in kept_set]
        if len(masked_list) > 0:
            masked_edges = torch.tensor(masked_list, device=device).t()
            pos_edges = edge_index_orig
            pred_pos = self.mask_predictor(z_q, pos_edges)
            pred_neg = self.mask_predictor(z_q, masked_edges)
            w_pos = (masked_edges.size(1) + pos_edges.size(1)) / (2 * pos_edges.size(1))
            w_neg = (masked_edges.size(1) + pos_edges.size(1)) / (2 * masked_edges.size(1))
            l_ban = - w_pos * torch.log(pred_pos + 1e-15).mean() - w_neg * torch.log(1 - pred_neg + 1e-15).mean()
        else:
            l_ban = torch.tensor(0.0, device=device)

        total_loss = l1 + l2 + l_ban + 0.1 * l_fg + 0.01 * l_pur + 0.001 * l_cluster

        self._momentum_update()

        return total_loss, {'l1': l1.item(), 'l2': l2.item(), 'l_ban': l_ban.item(),
                            'l_fg': l_fg.item(), 'l_pur': l_pur.item(), 'l_cluster': l_cluster.item()}

    @torch.no_grad()
    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        x_ban = self.feat_mask(x, edge_index)
        z = self.encoder_q(x_ban, edge_index)
        return z

    @torch.no_grad()
    def test_step(self, data, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self.get_embeddings(data)
        pos_pred = self.inner_product_decoder(z, pos_edge_index).sigmoid()
        neg_pred = self.inner_product_decoder(z, neg_edge_index).sigmoid()
        pred = torch.cat([pos_pred, neg_pred])
        label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        auc = roc_auc_score(label.cpu(), pred.cpu())
        ap = average_precision_score(label.cpu(), pred.cpu())
        return auc, ap

    @torch.no_grad()
    def test_step_ogb(self, data, evaluator, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self.get_embeddings(data)
        pos_pred = self.inner_product_decoder(z, pos_edge_index).sigmoid()
        neg_pred = self.inner_product_decoder(z, neg_edge_index).sigmoid()
        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})[f"hits@{K}"]
            results[f"Hits@{K}"] = hits
        return results