import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import numpy as np
import pandas as pd
import random
import argparse
import gc

from model_cgr import CrossFusion_DHCL_Net
from utils import (
    extract_edge_data_from_sparse_tensor, 
    load_kfold_data, 
    load_label_single, 
    stratified_kfold_split,
    processingIncidenceMatrix,
    create_dynamic_prototypes,
    calculate_contrastive_loss
)


# Hyperparameter and setting
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset (STRING, CPDB, BioGRID)')
parser.add_argument('cancerType', type=str, help='Types of cancer (pan-cancer, kirc, brca, ... )')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=int, default=0, help='GPU device ID (if available)')
parser.add_argument('--margin', type=float, default=1.0, help='Margin value used in the triplet-based contrastive loss.')
args = parser.parse_args()


device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Loss Function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        loss = self.alpha * ((1 - pt) ** self.gamma) * BCE
        return loss.mean()
    

# Data input
dataPath = f"./Data/{args.dataset}"
print(dataPath)
# load new multi-omics feature 
data_x_df = pd.read_csv(dataPath + f'/multiomics_features_{args.dataset}.tsv', sep='\t', index_col=0)
data_x_df = data_x_df.dropna()
gene_list_master = data_x_df.index.tolist() # Master gene list
print(f"Master gene list created with {len(gene_list_master)} genes.")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
data_x = data_x[:,:48]

cancerType = args.cancerType.lower()
num_nodes = features_scaled.shape[0]

# --- Cancer type definitions ---
cancer_names = ['kirc', 'brca', 'prad', 'stad', 'hnsc', 'luad', 'thca', 
                'blca', 'esca', 'lihc', 'ucec', 'coad', 'lusc', 'cesc', 'kirp']
modalities = ['MF', 'METH', 'GE'] # Multi-omics modalities to be used

cancer_type_to_id = {name: i for i, name in enumerate(cancer_names)}
num_cancer_types = len(cancer_names)

if cancerType=='pan-cancer':
    data_x = data_x[:,:48]
    print("--- [INFO] Loading hyperparameters for Pan-Cancer ---")
else:
    cancerType_dict = {
                     'kirc':[0,16,32],
                     'brca':[1,17,33],
                     'prad':[3,19,35],
                     'stad':[4,20,36],
                     'hnsc':[5,21,37],
                     'luad':[6,22,38],
                     'thca':[7,23,39],
                     'blca':[8,24,40],
                     'esca':[9,25,41],
                     'lihc':[10,26,42],
                     'ucec':[11,27,43],
                     'coad':[12,28,44],
                     'lusc':[13,29,45],
                     'cesc':[14,30,46],
                     'kirp':[15,31,47]
                     }
    data_x = data_x[:, cancerType_dict[cancerType]]
    print(f"===== [INFO] Loading hyperparameters for Specific Cancer: {cancerType.upper()} =====")

node_features = data_x  # torch.Tensor, [N, 48]
omics_dim = node_features.shape[1]

# ESM-2 Embeddings (Intrinsic Properties)
print("Loading ESM-2 embeddings...")
esm_embeddings_dict = torch.load(f"{dataPath}/esm2_embeddings.pt")
esm_dim = next(iter(esm_embeddings_dict.values())).shape[0]
esm_features = torch.zeros(len(gene_list_master), esm_dim, device=device)
for i, gene in enumerate(gene_list_master):
    if gene in esm_embeddings_dict:
        esm_features[i] = esm_embeddings_dict[gene].to(device)
    else:
        # If ESM embedding is missing, leave as 0
        pass 
print(f"Aligned ESM-2 features created with shape: {esm_features.shape}")

ppiAdj = torch.load(dataPath+f'/{args.dataset}_ppi.pkl')
pathAdj = torch.load(dataPath+'/pathway_SimMatrix_filtered.pkl')
goAdj = torch.load(dataPath+'/GO_SimMatrix_filtered.pkl')

ppi_row, ppi_col, ppi_score = extract_edge_data_from_sparse_tensor(ppiAdj)
go_row, go_col, go_score = extract_edge_data_from_sparse_tensor(goAdj)
path_row, path_col, path_score = extract_edge_data_from_sparse_tensor(pathAdj)

num_nodes = ppiAdj.shape[0] # The shape of the sparse tensor is [num_nodes, num_nodes]
num_edges = ppi_score.shape[0]  # The values (scores) of the coalesced sparse tensor represent individual edges

# --- Causality Score (PageRank) Calculation ---
print("--- [INFO] Calculating Causal Scores (PageRank) on-the-fly ---")
# 1. Create NetworkX graph object
num_nodes = data_x.shape[0] # Total number of genes
print("Converting sparse PPI matrix to edge list...")
ppi_sparse_coalesced = ppiAdj.coalesce()
indices = ppi_sparse_coalesced.indices()
edge_list = indices.t().tolist()

G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edge_list)
print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# 2. Calculate PageRank
print("Calculating PageRank...")
pagerank_scores_dict = nx.pagerank(G, alpha=0.85)

# 3. Align PageRank scores into a tensor according to the master gene list order
causal_scores = torch.zeros(num_nodes, 1) # Create with shape [N, 1]
for node_idx, score in pagerank_scores_dict.items():
    if node_idx < num_nodes:
        causal_scores[node_idx, 0] = score

# 4. Normalize (Min-Max Scaling) and move to device
min_val = torch.min(causal_scores)
max_val = torch.max(causal_scores)
if max_val > min_val:
    causal_scores = (causal_scores - min_val) / (max_val - min_val)

causal_scores = causal_scores.to(device)
print(f"Causal scores created and moved to device. Shape: {causal_scores.shape}")
# --- Calculation End ---


# Create edge_index tensor (shape: [2, num_edges]) using torch.stack()
edge_indices = [
    torch.stack([ppi_row, ppi_col], dim=0).to(device),
    torch.stack([go_row, go_col], dim=0).to(device),
    torch.stack([path_row, path_col], dim=0).to(device)
]

edge_scores = [
    ppi_score.to(device),
    go_score.to(device),
    path_score.to(device)
]

num_views = len(edge_indices)

# Create edge_indices as a dictionary
edge_indices_dict = {
    'ppi': torch.stack([ppi_row, ppi_col], dim=0).to(device),
    'go': torch.stack([go_row, go_col], dim=0).to(device),
    'path': torch.stack([path_row, path_col], dim=0).to(device)
}

# Create edge_scores as a dictionary
edge_scores_dict = {
    'ppi': ppi_score.to(device),
    'go': go_score.to(device),
    'path': path_score.to(device)
}

# Load and create gene set incidence matrix
print("--- [INFO] Processing Gene Set Incidence Matrix ---")
msigdb_genelist = pd.read_csv('./Data/msigdb/geneList.csv', header=None)
msigdb_genelist = list(msigdb_genelist[0].values)
incidenceMatrix = processingIncidenceMatrix(msigdb_genelist, dataPath)
gene_set_matrix = torch.tensor(incidenceMatrix.values, dtype=torch.float32, device=device)
num_gene_sets = gene_set_matrix.shape[1]


    # embed_dim = 256
    # beta = 0.01
    # learning_rate = 3e-5
    # epochs = 120
    # num_heads = 8
    # dropout = 0.2

# Set hyperparameters
if cancerType == 'pan-cancer':
    print("--- [INFO] Loading hyperparameters for Pan-Cancer ---")
    embed_dim = 64
    beta = 0.01
    learning_rate = 7e-5
    epochs = 120
    num_heads = 4
    dropout = 0.3
else:
    print(f"===== [INFO] Loading hyperparameters for Specific Cancer: {cancerType.upper()} =====")
    embed_dim = 128
    beta = 0.05
    learning_rate = 5e-5
    epochs = 150
    num_heads = 4
    dropout = 0.2

print(f"Applied Hyperparameters: Learning Rate={learning_rate}, Epochs={epochs}, Heads={num_heads}, Dropout={dropout}")


# Model Train & Test
cross_val=10
AUC = np.zeros(shape=(cross_val))
AUPR = np.zeros(shape=(cross_val))
F1_SCORES = np.zeros(cross_val)

if cancerType != 'pan-cancer':
    num_folds = cross_val
    path = f"{dataPath}/dataset/specific-cancer/"
    label_new, label_pos, label_neg = load_label_single(path, cancerType, device)
    random.shuffle(label_pos)
    random.shuffle(label_neg)
    l = len(label_new)
    l1 = int(len(label_pos)/num_folds)
    l2 = int(len(label_neg)/num_folds)
    folds = stratified_kfold_split(label_pos, label_neg, l, l1, l2)
    Y = label_new

# 1. Check initial node features
print(f"Node Features (omics) shape: {node_features.shape}, Contains NaN: {torch.isnan(node_features).any()}")
print(f"ESM Features shape: {esm_features.shape}, Contains NaN: {torch.isnan(esm_features).any()}")
print(f"Gene Set Matrix shape: {gene_set_matrix.shape}, Contains NaN: {torch.isnan(gene_set_matrix).any()}")

# 2. Check graph structure
graph_names = ['PPI', 'GO', 'Pathway']
nan_found_in_scores = False
for i in range(num_views):
    edges = edge_indices[i]
    scores = edge_scores[i]
    print(f"Graph '{graph_names[i]}' - Edges shape: {edges.shape}, Scores shape: {scores.shape}")
    if torch.isnan(scores).any():
        print(f"   [CRITICAL] !!! NaN DETECTED IN '{graph_names[i]}' EDGE SCORES !!!")
        nan_found_in_scores = True
    else:
        print(f"   '{graph_names[i]}' Edge Scores are clean.")

# 3. Check causal scores
print(f"Causal Scores shape: {causal_scores.shape}, Contains NaN: {torch.isnan(causal_scores).any()}")
print("--- Sanity Check Complete ---\n")
if nan_found_in_scores:
    print("[FIX] Replacing NaN values in edge_scores with 0.0...")
    for i in range(num_views):
        edge_scores[i] = torch.nan_to_num(edge_scores[i], nan=0.0)
    print("[FIX] NaN replacement complete.")


print(f"----- {args.cancerType.upper()} 10-fold validation ------")
for fold in range(cross_val):
    print(f'--------- Fold {fold+1} Begin ---------')
    if args.cancerType == 'pan-cancer':
        fold_path = f"{dataPath}/10fold/fold_{fold+1}"
        train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, Y = load_kfold_data(fold_path, device)
    else:
        train_idx, valid_idx, test_idx, train_mask, val_mask, test_mask = folds[fold]
    
    model = CrossFusion_DHCL_Net(
        omics_dim=omics_dim,
        esm_dim=esm_dim,
        num_gene_sets=num_gene_sets,
        embed_dim=embed_dim,
        num_views=num_views,
        num_cancer_types=num_cancer_types,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    # Create prototypes at the start of the fold
    p_pos, p_neg = create_dynamic_prototypes(
        train_idx, Y, incidenceMatrix, model, device
    )
    # Detach prototypes from the computation graph to stop gradients
    p_pos = p_pos.detach()
    p_neg = p_neg.detach()
    
    # Now, map the clean tensors (with no grad_fn) to hyperbolic space
    p_pos_hyp = model.manifold.expmap0(p_pos)
    p_neg_hyp = model.manifold.expmap0(p_neg)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5
    )
    
    criterion_class = FocalLoss(alpha=1.0, gamma=1.5)
    
    cancer_type_id_tensor = None
    if args.cancerType.lower() in cancer_type_to_id:
        cancer_id = cancer_type_to_id[args.cancerType.lower()]
        cancer_type_id_tensor = torch.tensor([cancer_id], device=device)
        print(f'cancer_id : {cancer_id}')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
    
        logits, hyp_embeds = model(
            omics_features=node_features,
            esm_features=esm_features,
            gene_set_matrix=gene_set_matrix,
            edge_indices=edge_indices,
            edge_scores=edge_scores,
            cancer_type_id=cancer_type_id_tensor,
            causal_scores=causal_scores
        )
        # Calculate classification loss
        loss_cls = criterion_class(logits[train_idx], Y[train_idx]) 
        # Calculate contrastive loss
        contrastive_loss = calculate_contrastive_loss(
            hyp_embeds[train_idx], Y[train_idx], 
            p_pos_hyp, p_neg_hyp, model.manifold, args.margin
        )
        loss = loss_cls + beta * contrastive_loss
        
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            
            pred_logits, _ = model(
                omics_features=node_features,
                esm_features=esm_features,
                gene_set_matrix=gene_set_matrix,
                edge_indices=edge_indices,
                edge_scores=edge_scores,
                cancer_type_id=cancer_type_id_tensor,
                causal_scores=causal_scores
            )

            pred_probs = torch.sigmoid(pred_logits)

            val_auc = roc_auc_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            val_aupr = average_precision_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            print(f"[Fold {fold+1}] Epoch {epoch+1} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")
    
    model.eval()
    with torch.no_grad():
        
        final_logits, _ = model(
            omics_features=node_features,
            esm_features=esm_features,
            gene_set_matrix=gene_set_matrix,
            edge_indices=edge_indices,
            edge_scores=edge_scores,
            cancer_type_id=cancer_type_id_tensor,
            causal_scores=causal_scores
        )

        final_probs = torch.sigmoid(final_logits)
        pred_labels = (final_probs > 0.5).float()

        AUC[fold] = roc_auc_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        AUPR[fold] = average_precision_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        F1_SCORES[fold] = f1_score(Y[test_idx].cpu(), pred_labels[test_idx].cpu())
    
    print(f"Fold {fold+1} Results — AUC: {AUC[fold]:.3f}, AUPR: {AUPR[fold]:.3f}, F1: {F1_SCORES[fold]:.3f}")

    with open(f"./final_results_{args.dataset}_{args.cancerType.upper()}.txt", "a") as result_file:
        result_file.write(f"Fold {fold+1}: AUC={AUC[fold]:.3f}, AUPR={AUPR[fold]:.3f}, F1-score={F1_SCORES[fold]:.3f}\n")
    
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
print("========== Final 10-Fold Results ==========")
print(f"Mean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}")
print(f"Mean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}")
print(f"Mean F1: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}")

with open(f"./final_results_{args.dataset}_{args.cancerType.upper()}.txt", "a", encoding="utf-8") as result_file:
    result_file.write(f"\nFinal Results:\nMean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}\nMean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}\nMean F1-score: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}\n\n")
    
    