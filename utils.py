import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F

def processingIncidenceMatrix(geneList, dataPath):
    feature_genename_file = f'{dataPath}/feature_genename.txt'  # feature_genename.txt - feature gene names of PPI network
    filtered_geneList = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()

    print(f"Original geneList size: {len(geneList)} → Filtered size: {len(filtered_geneList)}")
    
    ids = ['c2','c5']
    # incidenceMatrix = pd.DataFrame(index= geneList)
    incidenceMatrix = pd.DataFrame(index=filtered_geneList)
    for id in ids:
        geneSetNameList = pd.read_csv('./Data/msigdb/'+id+'Name.txt',sep='\t',header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        z=0
        idList = list()
        for name in geneSetNameList:
            if(id=='c2'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q or 'CARCINOMA' in q or 'LEUKEMIA' in q or 'SARCOMA' in q):
                    pass
                else:
                    idList.append(z)
            elif(name[:2]=='HP'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q or 'CARCINOMA' in q or 'LEUKEMIA' in q or 'SARCOMA' in q):
                    pass
                else:
                    idList.append(z)
            else:
                idList.append(z)
            z=z+1
        genesetData = sp.load_npz('./Data/msigdb/'+id+'_GenesetsMatrix.npz')
        
        incidenceMatrixTemp = pd.DataFrame(data=genesetData.A, index=geneList)
        incidenceMatrixTemp = incidenceMatrixTemp.loc[geneList]  # Ensure that the original geneList index is used
        incidenceMatrixTemp = incidenceMatrixTemp.reindex(index=filtered_geneList, fill_value=0)  # Missing genes filled with 0
        incidenceMatrixTemp = incidenceMatrixTemp.iloc[:, idList]
        
        # Merge the new data into the overall incidence matrix
        incidenceMatrix = pd.concat([incidenceMatrix, incidenceMatrixTemp], axis=1)

    # column indexing with numbers
    incidenceMatrix.columns = np.arange(incidenceMatrix.shape[1])
    print(f"Final incidenceMatrix shape: {incidenceMatrix.shape}")
    
    return incidenceMatrix


def extract_edge_data_from_sparse_tensor(sparse_tensor):
    """PyTorch 희소 텐서에서 row, col 인덱스와 score를 추출합니다."""
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    row = indices[0]
    col = indices[1]
    score = sparse_tensor.values()
    return row, col, score


"""Load the fold of STRING/CPDB pan-cancer dataset"""
def load_kfold_data(fold_path, device):
    # Load indices
    train_idx = np.loadtxt(f"{fold_path}/train.txt", dtype=int)
    valid_idx = np.loadtxt(f"{fold_path}/valid.txt", dtype=int)
    test_idx = np.loadtxt(f"{fold_path}/test.txt", dtype=int)
    
    # Load masks
    train_mask = torch.tensor(np.loadtxt(f"{fold_path}/train_mask.txt", dtype=bool), device=device)
    valid_mask = torch.tensor(np.loadtxt(f"{fold_path}/valid_mask.txt", dtype=bool), device=device)
    test_mask = torch.tensor(np.loadtxt(f"{fold_path}/test_mask.txt", dtype=bool), device=device)
    
    # Load labels
    labels = torch.tensor(np.loadtxt(f"{fold_path}/labels.txt"), dtype=torch.float32, device=device)
    
    return train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, labels


"""Load the fold of STRING/CPDB specific-cancer dataset"""
def load_label_single(path, cancerType, device):
    label = np.loadtxt(path + "label_file-P-"+cancerType+".txt")
    Y = torch.tensor(label, dtype=torch.float32, device=device)
    label_pos = np.loadtxt(path + "pos-"+cancerType+".txt", dtype=int)
    label_neg = np.loadtxt(path + "pan-neg.txt", dtype=int)
    return Y, label_pos, label_neg

def stratified_kfold_split(pos_label, neg_label, l, l1, l2):
    folds = []
    for i in range(10):
        pos_test = list(pos_label[i * l1:(i + 1) * l1])
        pos_train = list(set(pos_label) - set(pos_test))
        neg_test = list(neg_label[i * l2:(i + 1) * l2])
        neg_train = list(set(neg_label) - set(neg_test))

        val_size_pos = len(pos_train) // 8
        val_size_neg = len(neg_train) // 8

        pos_val = list(pos_train[:val_size_pos])
        pos_train_final = list(pos_train[val_size_pos:])
        neg_val = list(neg_train[:val_size_neg])
        neg_train_final = list(neg_train[val_size_neg:])

        train_idx = sorted(pos_train_final + neg_train_final)
        val_idx = sorted(pos_val + neg_val)
        test_idx = sorted(pos_test + neg_test)

        # 마스크 생성
        indexs1 = [False] * l
        indexs2 = [False] * l
        indexs3 = [False] * l
        for j in train_idx:
            indexs1[j] = True
        for j in val_idx:
            indexs2[j] = True
        for j in test_idx:
            indexs3[j] = True

        train_mask = torch.from_numpy(np.array(indexs1))
        val_mask = torch.from_numpy(np.array(indexs2))
        test_mask = torch.from_numpy(np.array(indexs3))

        folds.append((train_idx, val_idx, test_idx, train_mask, val_mask, test_mask))

    return folds

def create_dynamic_prototypes(train_idx, label_tensor, incidence_matrix, model, device):
    """Create 'driver' and 'non-driver' functional prototypes using training labels."""
    train_labels = label_tensor[train_idx]
    train_genes = incidence_matrix.index[train_idx]
    
    positive_genes = train_genes[(train_labels == 1).cpu()]
    # Calculate the sum of gene sets to which positive genes in the training data belong
    positive_matrix_sum = incidence_matrix.loc[positive_genes].sum()
    
    # Consider gene sets containing 3 or more driver genes as 'positive functions'
    positive_func_indices = torch.tensor(
        np.where(positive_matrix_sum >= 3)[0], device=device
    )
    # Consider gene sets with no driver genes as 'negative functions'
    negative_func_indices = torch.tensor(
        np.where(positive_matrix_sum == 0)[0], device=device
    )
    
    # Get the gene set embedding table from the model
    set_embeddings = model.func_encoder.gene_set_embedding.weight
    
    # Create prototypes by averaging the embeddings of each functional group
    p_pos = set_embeddings[positive_func_indices].mean(dim=0)
    p_neg = set_embeddings[negative_func_indices].mean(dim=0)
    
    return p_pos, p_neg

def calculate_contrastive_loss(hyp_embeds, labels, p_pos_hyp, p_neg_hyp, manifold, margin=1.0):
    """Calculate contrastive loss based on Triplet Loss in hyperbolic space."""
    positive_anchors = hyp_embeds[labels == 1]
    if positive_anchors.size(0) == 0:
        return 0.0 # If there are no positive samples in the batch
    
    dist_pos = manifold.dist(positive_anchors, p_pos_hyp.expand_as(positive_anchors))
    dist_neg = manifold.dist(positive_anchors, p_neg_hyp.expand_as(positive_anchors))
    loss = F.relu(dist_pos**2 - dist_neg**2 + margin).mean()
    return loss