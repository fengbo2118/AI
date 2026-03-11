import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce, degree, to_undirected
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.loader import NeighborLoader
from torch_scatter import scatter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


def get_dataset(root, name, transform=None):
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root, name, transform=transform)
        return dataset[0]
    elif name.lower() == 'chameleon':
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root, name='chameleon', transform=transform)
        return dataset[0]
    else:
        raise ValueError(f'Unknown dataset {name}')


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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embedding(self, x, edge_index):
        return self.forward(x, edge_index)


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
        self.convs.append(
            GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))))

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x


class DPVGAE_NodeClas(nn.Module):
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

        return total_loss

    @torch.no_grad()
    def get_embeddings(self, data):
        x, edge_index = data.x, data.edge_index
        x_ban = self.feat_mask(x, edge_index)
        z = self.encoder_q(x_ban, edge_index)
        return z

    @torch.no_grad()
    def test_step(self, data, pos_edge_index, neg_edge_index, batch_size=2 ** 16):
        self.eval()
        z = self.get_embeddings(data)
        pos_pred = self.inner_product_decoder(z, pos_edge_index).sigmoid()
        neg_pred = self.inner_product_decoder(z, neg_edge_index).sigmoid()
        pred = torch.cat([pos_pred, neg_pred])
        label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(label.cpu(), pred.cpu())
        ap = average_precision_score(label.cpu(), pred.cpu())
        return auc, ap


def train_linkpred(model, splits, args, device="cpu"):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_valid = 0
    batch_size = args.batch_size

    train_data = splits['train'].to(device)
    valid_data = splits['valid'].to(device)
    test_data = splits['test'].to(device)

    for epoch in tqdm(range(1, 1 + args.epochs)):
        model.train()
        optimizer.zero_grad()
        loss = model(train_data)
        loss.backward()
        if args.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        if epoch % args.eval_period == 0:
            valid_auc, valid_ap = model.test_step(valid_data,
                                                  valid_data.pos_edge_label_index,
                                                  valid_data.neg_edge_label_index,
                                                  batch_size=batch_size)
            if valid_auc > best_valid:
                best_valid = valid_auc
                torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(torch.load(args.save_path))
    test_auc, test_ap = model.test_step(test_data,
                                        test_data.pos_edge_label_index,
                                        test_data.neg_edge_label_index,
                                        batch_size=batch_size)

    print(f'Link Prediction Pretraining Results:\n'
          f'AUC: {test_auc:.2%}',
          f'AP: {test_ap:.2%}')
    return test_auc, test_ap


def train_nodeclas(model, data, args, device='cpu'):
    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    if args.dataset in {'arxiv', 'products', 'mag'}:
        batch_size = 4096
    else:
        batch_size = 512

    train_loader = DataLoader(data.train_mask.nonzero().squeeze(), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data.test_mask.nonzero().squeeze(), batch_size=20000)
    val_loader = DataLoader(data.val_mask.nonzero().squeeze(), batch_size=20000)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.get_embeddings(data)

    if args.l2_normalize:
        embedding = F.normalize(embedding, p=2, dim=1)

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    print('Start Training (Node Classification)...')
    results = []

    for run in range(1, args.runs + 1):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(),
                                     lr=0.01,
                                     weight_decay=args.nodeclas_weight_decay)

        best_val_metric = test_metric = 0
        for epoch in tqdm(range(1, 101), desc=f'Training on runs {run}...'):
            clf.train()
            for nodes in train_loader:
                optimizer.zero_grad()
                loss_fn(clf(embedding[nodes]), y[nodes]).backward()
                optimizer.step()

            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
        results.append(best_test_metric)
        print(f'Runs {run}: accuracy {best_test_metric:.2%}')

    print(f'Node Classification Results ({args.runs} runs):\n'
          f'Accuracy: {np.mean(results):.2%} ± {np.std(results):.2%}')


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Cora")
parser.add_argument("--mask", default="Path")
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument("--layer", default="gcn")
parser.add_argument("--encoder_activation", default="elu")
parser.add_argument('--encoder_channels', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--decoder_channels', type=int, default=32)
parser.add_argument('--encoder_layers', type=int, default=2)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--encoder_dropout', type=float, default=0.8)
parser.add_argument('--decoder_dropout', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=0.)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--grad_norm', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=2 ** 16)
parser.add_argument("--start", default="node")
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--bn', action='store_true')
parser.add_argument('--l2_normalize', action='store_true')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--eval_period', type=int, default=30)
parser.add_argument("--save_path", default="DPVGAE-NodeClas.pt")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--full_data', action='store_true')
parser.add_argument('--momentum', type=float, default=0.999)
parser.add_argument('--topo_threshold', type=float, default=0.5)
parser.add_argument('--recovery_rate', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--alpha_temp', type=float, default=0.5)
parser.add_argument('--num_clusters', type=int, default=7)

args = parser.parse_args()
print(tab_printer(args))
set_seed(args.seed)

if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

transform = T.Compose([T.ToUndirected(), T.ToDevice(device)])
root = 'data/'
data = get_dataset(root, args.dataset, transform=transform)

train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=False)(data)
if args.full_data:
    splits = dict(train=data, valid=val_data, test=test_data)
else:
    splits = dict(train=train_data, valid=val_data, test=test_data)

model = DPVGAE_NodeClas(data.num_features, data.num_nodes, args).to(device)

train_linkpred(model, splits, args, device=device)
train_nodeclas(model, data, args, device=device)