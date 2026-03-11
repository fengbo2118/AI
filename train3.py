import argparse
from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce, degree, to_undirected
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from tqdm.auto import tqdm
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def tab_printer(args):
    args = vars(args)
    max_len = max([len(k) for k in args.keys()])
    rows = []
    for k, v in args.items():
        rows.append(f'{k}{" " * (max_len - len(k))} : {v}')
    return '\n'.join(rows)

class TopologyAdaptiveFilter:
    def __init__(self, threshold=0.5, recovery_rate=0.7, total_steps=200, beta=1.0):
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

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

class DPVGAE_OGB(nn.Module):
    def __init__(self, num_nodes, num_features, args):
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

        if num_features == 0:
            self.node_emb = nn.Embedding(num_nodes, args.encoder_channels)
            in_dim = args.encoder_channels
        else:
            self.node_emb = None
            in_dim = num_features

        self.encoder_q = GNNEncoder(in_dim, args.encoder_channels, args.hidden_channels,
                                    num_layers=args.encoder_layers, dropout=args.encoder_dropout)
        self.encoder_k = GNNEncoder(in_dim, args.encoder_channels, args.hidden_channels,
                                    num_layers=args.encoder_layers, dropout=args.encoder_dropout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.inner_product_decoder = InnerProductDecoder()
        self.feature_decoder = FeatureDecoder(args.hidden_channels, in_dim, args.decoder_channels)
        self.mask_predictor = MaskPredictor(args.hidden_channels, args.decoder_channels)

        self.gin = GINEncoder(args.hidden_channels, args.decoder_channels, args.decoder_channels, num_layers=2)

        self.num_clusters = args.num_clusters
        self.cluster_centers = nn.Parameter(torch.randn(args.num_clusters, args.hidden_channels))

        self.momentum = args.momentum

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, data):
        if self.node_emb is not None:
            x = self.node_emb.weight
        else:
            x = data.x
        edge_index_orig = data.edge_index

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
            l_pur = torch.tensor(0.0, device=x.device)
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
        neg_edge_index = torch.randint(0, self.num_nodes, (2, edge_index_orig.size(1)), device=x.device)
        neg_logits = self.inner_product_decoder(z_q, neg_edge_index)
        neg_loss = -F.logsigmoid(-neg_logits).mean()
        l1 = pos_loss + neg_loss

        x_hat = self.feature_decoder(z_q)
        l2 = F.mse_loss(x_hat, x)

        kept_set = set(map(tuple, edge_index_pur.t().tolist()))
        all_set = set(map(tuple, edge_index_orig.t().tolist()))
        masked_list = [e for e in all_set if e not in kept_set]
        if len(masked_list) > 0:
            masked_edges = torch.tensor(masked_list, device=x.device).t()
            pos_edges = edge_index_orig
            pred_pos = self.mask_predictor(z_q, pos_edges)
            pred_neg = self.mask_predictor(z_q, masked_edges)
            w_pos = (masked_edges.size(1) + pos_edges.size(1)) / (2 * pos_edges.size(1))
            w_neg = (masked_edges.size(1) + pos_edges.size(1)) / (2 * masked_edges.size(1))
            l_ban = - w_pos * torch.log(pred_pos + 1e-15).mean() - w_neg * torch.log(1 - pred_neg + 1e-15).mean()
        else:
            l_ban = torch.tensor(0.0, device=x.device)

        total_loss = l1 + l2 + l_ban + 0.1 * l_fg + 0.01 * l_pur + 0.001 * l_cluster

        self._momentum_update()

        return total_loss

    @torch.no_grad()
    def get_embeddings(self, data):
        if self.node_emb is not None:
            x = self.node_emb.weight
        else:
            x = data.x
        x_ban = self.feat_mask(x, data.edge_index)
        z = self.encoder_q(x_ban, data.edge_index)
        return z

    @torch.no_grad()
    def test_step_ogb(self, data, evaluator, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self.get_embeddings(data)
        pos_pred = self.inner_product_decoder(z, pos_edge_index)
        neg_pred = self.inner_product_decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred])
        y_true = torch.cat([torch.ones(pos_pred.size(0), dtype=torch.int),
                            torch.zeros(neg_pred.size(0), dtype=torch.int)])
        input_dict = {"y_pred_pos": y_pred[:pos_pred.size(0)],
                      "y_pred_neg": y_pred[pos_pred.size(0):],
                      "evaluator": evaluator}
        return evaluator.eval(input_dict)

def train_linkpred(model, splits, args, device="cpu"):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid = 0
    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)
    monitor = 'Hits@50'
    evaluator = Evaluator(name=args.dataset)

    for epoch in tqdm(range(1, 1 + args.epochs)):
        model.train()
        optimizer.zero_grad()
        loss = model(train_data)
        loss.backward()
        if args.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        if epoch % args.eval_period == 0:
            valid_results = model.test_step_ogb(valid_data, evaluator,
                                                valid_data.pos_edge_label_index,
                                                valid_data.neg_edge_label_index,
                                                batch_size=args.batch_size)
            test_results = model.test_step_ogb(test_data, evaluator,
                                               test_data.pos_edge_label_index,
                                               test_data.neg_edge_label_index,
                                               batch_size=args.batch_size)
            if valid_results[monitor] > best_valid:
                best_valid = valid_results[monitor]
                torch.save(model.state_dict(), args.save_path)
            print(f"Epoch {epoch} - Hits@20: {test_results['Hits@20']:.4f}, Hits@50: {test_results['Hits@50']:.4f}, Hits@100: {test_results['Hits@100']:.4f}")

    model.load_state_dict(torch.load(args.save_path))
    results = model.test_step_ogb(test_data, evaluator,
                                  test_data.pos_edge_label_index,
                                  test_data.neg_edge_label_index,
                                  batch_size=args.batch_size)
    return results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ogbl-collab")
parser.add_argument("--mask", default="Path")
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--bn', action='store_true')
parser.add_argument("--layer", default="gcn")
parser.add_argument("--encoder_activation", default="elu")
parser.add_argument('--encoder_channels', type=int, default=256)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--decoder_channels', type=int, default=64)
parser.add_argument('--encoder_layers', type=int, default=2)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--encoder_dropout', type=float, default=0.3)
parser.add_argument('--decoder_dropout', type=float, default=0.3)
parser.add_argument('--alpha', type=float, default=0.003)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=2**16)
parser.add_argument("--start", default="edge")
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--eval_period', type=int, default=10)
parser.add_argument("--save_path", default="DPVGAE-OGB.pt")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--momentum', type=float, default=0.999)
parser.add_argument('--topo_threshold', type=float, default=0.5)
parser.add_argument('--recovery_rate', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--alpha_temp', type=float, default=0.5)
parser.add_argument('--num_clusters', type=int, default=10)
args = parser.parse_args()

print(tab_printer(args))
set_seed(args.seed)
device = f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"

transform = T.Compose([T.ToDevice(device)])
root = 'data/'
print('Loading Data...')
if args.dataset in {'ogbl-collab'}:
    dataset = PygLinkPropPredDataset(name=args.dataset, root=root)
    data = transform(dataset[0])
    if hasattr(data, 'edge_weight'):
        del data.edge_weight
    if hasattr(data, 'edge_year'):
        del data.edge_year
else:
    raise ValueError(args.dataset)

split_edge = dataset.get_edge_split()
args.year = 2010
if args.year > 0:
    year_mask = split_edge['train']['year'] >= args.year
    split_edge['train']['edge'] = split_edge['train']['edge'][year_mask]
    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    print(f"{1 - year_mask.float().mean():.2%} edges dropped by year {args.year}.")

train_data, val_data, test_data = copy(data), copy(data), copy(data)
args.val_as_input = True
if args.val_as_input:
    full_edge_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
    full_edge_index = to_undirected(full_edge_index)
    train_data.edge_index = full_edge_index
    val_data.edge_index = full_edge_index
    test_data.edge_index = full_edge_index
    train_data.pos_edge_label_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge']], dim=0).t()
else:
    train_data.pos_edge_label_index = split_edge['train']['edge'].t()

val_data.pos_edge_label_index = split_edge['valid']['edge'].t()
val_data.neg_edge_label_index = split_edge['valid']['edge_neg'].t()
test_data.pos_edge_label_index = split_edge['test']['edge'].t()
test_data.neg_edge_label_index = split_edge['test']['edge_neg'].t()

splits = dict(train=train_data, valid=val_data, test=test_data)

num_features = data.num_features if hasattr(data, 'num_features') else 0
model = DPVGAE_OGB(data.num_nodes, num_features, args).to(device)

hit_20, hit_50, hit_100 = [], [], []
for run in range(1, args.runs + 1):
    hit = train_linkpred(model, splits, args, device=device)
    hit_20.append(hit['Hits@20'])
    hit_50.append(hit['Hits@50'])
    hit_100.append(hit['Hits@100'])
    print(f"Run {run} - Hits@20: {hit['Hits@20']:.4f}, Hits@50: {hit['Hits@50']:.4f}, Hits@100: {hit['Hits@100']:.4f}")

print(f'Link Prediction Results ({args.runs} runs):')
print(f'Hits@20: {np.mean(hit_20):.4f} ± {np.std(hit_20):.4f}')
print(f'Hits@50: {np.mean(hit_50):.4f} ± {np.std(hit_50):.4f}')
print(f'Hits@100: {np.mean(hit_100):.4f} ± {np.std(hit_100):.4f}')