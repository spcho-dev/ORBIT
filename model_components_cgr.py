import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops


class HyperbolicGCNLayer(MessagePassing):
    """Hyperbolic GCN Layer based on PyG's MessagePassing."""
    def __init__(self, in_features, out_features, manifold, dropout=0.3):
        super(HyperbolicGCNLayer, self).__init__(aggr='add')
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.agg_act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        identity = x
        # Map from manifold to tangent space
        x_tan = self.manifold.logmap0(x)
        # Apply linear transformation in tangent space
        x_linear = self.linear(x_tan)
        # Map back to manifold
        x_hyp_transformed = self.manifold.expmap0(x_linear)
        # Add self-loops for GCN behavior
        edge_index_looped, edge_weight_looped = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
        )
        # 4. Propagate messages
        h_out = self.propagate(edge_index_looped, x=x_hyp_transformed, edge_weight=edge_weight_looped)
        # 5. Apply residual connection using Mobius addition
        output = self.manifold.mobius_add(h_out, identity)
        return output
    
    def message(self, x_j, edge_weight):
        # Map neighbor nodes (x_j) to tangent space to create messages
        msg = self.manifold.logmap0(x_j)
        
        if edge_weight is not None:
            # Apply edge weights in tangent space
            return edge_weight.view(-1, 1) * msg
        return msg

    def update(self, aggr_out):
        # aggr_out is the aggregated message in the tangent space
        aggr_out_act = self.agg_act(aggr_out)
        aggr_out_dropped = F.dropout(aggr_out_act, p=self.dropout, training=self.training)
        
        # Map the final aggregated/updated features back to the manifold
        return self.manifold.expmap0(aggr_out_dropped)
    
    
class GraphRewiringNetwork(nn.Module):
    """A network to dynamically calculate edge weights (rewire the graph)"""
    def __init__(self, embed_dim, cancer_embed_dim):
        super(GraphRewiringNetwork, self).__init__()
        
        # Input features: h_i, h_j, cancer_emb, c_i, c_j, c_i - c_j
        # (embed_dim*2) + cancer_embed_dim + 3 (causal features)
        input_dim = embed_dim * 2 + cancer_embed_dim + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2, 1), nn.Sigmoid()
        )

    def forward(self, h, edge_index, cancer_embedding, causal_scores):
        h_i, h_j = h[edge_index[0]], h[edge_index[1]]
        
        # Extract causal scores for nodes in each edge
        c_i = causal_scores[edge_index[0]]
        c_j = causal_scores[edge_index[1]]
        
        # Provide explicit directional info via score difference
        causal_diff = c_i - c_j
        
        num_edges = edge_index.shape[1]
        cancer_embedding_expanded = cancer_embedding.expand(num_edges, -1)
        
        # Concatenate all features for the edge MLP
        edge_features = torch.cat([h_i, h_j, cancer_embedding_expanded, c_i, c_j, causal_diff], dim=-1)
        
        # Calculate new edge weights (scores)
        new_edge_weights = self.mlp(edge_features).squeeze(-1)
        return new_edge_weights
    
    
class HyperTopoGML_Backbone(nn.Module):
    """The main GNN backbone, integrating hyperbolic GCN layers and graph rewiring."""
    def __init__(self, in_features, embed_dim, num_views=3, c=1.0, 
                 dropout=0.3, num_heads=4, num_cancer_types=16):
        super(HyperTopoGML_Backbone, self).__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.embed_dim = embed_dim
        self.num_views = num_views
        
        cancer_embed_dim = embed_dim // 4
        self.cancer_embedding = nn.Embedding(num_cancer_types, cancer_embed_dim)
        
        # A separate rewiring network for each graph view
        self.rewiring_networks = nn.ModuleList([GraphRewiringNetwork(embed_dim, cancer_embed_dim) for _ in range(num_views)])
        
        # Two layers of hyperbolic GCN encoders for each view
        self.hyp_encoders_1 = nn.ModuleList([
            HyperbolicGCNLayer(embed_dim, embed_dim, self.manifold, dropout) for _ in range(num_views)
        ])
        self.hyp_encoders_2 = nn.ModuleList([
            HyperbolicGCNLayer(embed_dim, embed_dim, self.manifold, dropout) for _ in range(num_views)
        ])
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gating_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid()
        )

    def forward(self, x_proj, edge_indices, edge_scores, cancer_type_id, causal_scores):
        view_outputs = []
        hyp_embeddings_list = []
        
        context_embedding = None
        if cancer_type_id is not None:
            # Get the specific cancer type embedding
            context_embedding = self.cancer_embedding(cancer_type_id)

        for i in range(self.num_views):
            current_edge_weights = edge_scores[i]
            
            if context_embedding is not None:
                # Calculate dynamic, context-aware weights
                dynamic_weights = self.rewiring_networks[i](x_proj, edge_indices[i], context_embedding, causal_scores)
                # Rewire graph by multiplying original weights with dynamic weights
                current_edge_weights = current_edge_weights * dynamic_weights
            
            # Project initial features to manifold
            x_hyp_initial = self.manifold.expmap0(x_proj)
            
            # Pass through two layers of Hyperbolic GCN
            h_hyp_layer1 = self.hyp_encoders_1[i](x_hyp_initial, edge_indices[i], current_edge_weights)
            h_hyp_final = self.hyp_encoders_2[i](h_hyp_layer1, edge_indices[i], current_edge_weights)

            hyp_embeddings_list.append(h_hyp_final)
            
            # Project to tangent space for Euclidean operations (e.g., concatenation)
            h_tan = self.manifold.logmap0(h_hyp_final)
            view_outputs.append(h_tan)
            
        # Concatenate tangent space representations from all views
        final_tan_concat = torch.cat(view_outputs, dim=1)
        
        return final_tan_concat, hyp_embeddings_list
    
    
class FunctionalEncoder(nn.Module):
    """Encodes gene functional information using an attention-based weighted sum of gene set embeddings."""
    def __init__(self, num_gene_sets, embed_dim):
        super(FunctionalEncoder, self).__init__()
        self.gene_set_embedding = nn.Embedding(num_gene_sets, embed_dim)
        
        hidden_dim = max(num_gene_sets // 16, 256)
        self.attention_mlp = nn.Sequential(
            nn.Linear(num_gene_sets, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_gene_sets)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gene_set_matrix):
        # gene_set_matrix (Incidence Matrix): [N, M] (N=genes, M=gene_sets)
        
        # 1. Calculate attention scores for each gene set per gene
        # Input: [N, M], Output: [N, M]
        attention_scores = self.attention_mlp(gene_set_matrix.float())
        
        # 2. Mask scores for sets the gene does not belong to (using incidence matrix)
        attention_weights = self.softmax(attention_scores.masked_fill(gene_set_matrix == 0, -1e9))
        
        # 3. Compute weighted sum of gene set embeddings
        # [N, M] @ [M, D] -> [N, D]
        h_func = torch.matmul(attention_weights, self.gene_set_embedding.weight)
        
        return h_func
    
    
class CrossAttentionLayer(nn.Module):
    """Refines features of one modality (Query) using other modalities (Key/Value)."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv_context):
        # query: [N, D]
        # kv_context: [N, K*D] (K = number of other sources)
        
        # MultiheadAttention expects [N, L, D] (L=sequence length)
        query = query.unsqueeze(1) # [N, 1, D]
        
        # Reshape kv_context to [N, K, D]
        num_kv = kv_context.size(1) // query.size(2)
        kv = kv_context.view(query.size(0), num_kv, query.size(2))
        
        # Q=[N, 1, D], K=[N, K, D], V=[N, K, D]
        attn_output, _ = self.attn(query, kv, kv)
        
        return self.norm(query.squeeze(1) + self.dropout(attn_output.squeeze(1)))


class CrossModalityFuser(nn.Module):
    """
    Takes three information sources (h_omics, h_esm, h_func), 
    refines them with cross-attention, and fuses them with interpretable weights.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossModalityFuser, self).__init__()
        self.embed_dim = embed_dim
        
        # Cross-attention layers for refining each modality
        self.refine_omics = CrossAttentionLayer(embed_dim, num_heads, dropout)
        self.refine_esm = CrossAttentionLayer(embed_dim, num_heads, dropout)
        self.refine_func = CrossAttentionLayer(embed_dim, num_heads, dropout)
        
        # Attention gate for final fusion of refined features
        gate_input_dim = embed_dim * 3
        self.aggregation_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, h_omics, h_esm, h_func):
        # 1. Cross-Modality Refinement
        h_omics_refined = self.refine_omics(h_omics, torch.cat([h_esm, h_func], dim=1))
        h_esm_refined = self.refine_esm(h_esm, torch.cat([h_omics, h_func], dim=1))
        h_func_refined = self.refine_func(h_func, torch.cat([h_omics, h_esm], dim=1))
        
        # 2. Interpretable Weighted Fusion
        gate_input = torch.cat([h_omics_refined, h_esm_refined, h_func_refined], dim=1)
        
        # Use detach() for stable gate weight learning
        # This prevents gradients from flowing back from the weights to the features
        attention_weights = F.softmax(self.aggregation_gate(gate_input.detach()), dim=1).unsqueeze(-1)
        
        # Stack refined embeddings for weighted sum
        all_refined_embeds = torch.stack([h_omics_refined, h_esm_refined, h_func_refined], dim=1) # [N, 3, D]
        
        h_unified = torch.sum(attention_weights * all_refined_embeds, dim=1) # [N, D]
        
        # Return weights for interpretation
        return h_unified, attention_weights.squeeze(-1)