# File Name: CrossFusion_DHCL_Net.py

import torch
import torch.nn as nn
import geoopt
from model_components_cgr import (
    HyperTopoGML_Backbone,
    FunctionalEncoder,
    CrossModalityFuser
)

class CrossFusion_DHCL_Net(nn.Module):
    def __init__(self, omics_dim, esm_dim, num_gene_sets, embed_dim, num_views=3, num_cancer_types=16, num_heads=3, dropout=0.3):
        super(CrossFusion_DHCL_Net, self).__init__()
        
        # --- Step 1: Multi-Modal Encoders ---
        self.omics_encoder = nn.Linear(omics_dim, embed_dim)
        self.esm_encoder = nn.Linear(esm_dim, embed_dim)
        self.func_encoder = FunctionalEncoder(num_gene_sets, embed_dim)
        
        # --- Step 2: Cross-Modality Fusion ---
        self.fuser = CrossModalityFuser(embed_dim, num_heads, dropout)
        
        # --- Step 3: GNN Backbone (HyperTopoGML) ---
        # GNN backbone's in_features is now embed_dim (from h_unified)
        self.gnn_backbone = HyperTopoGML_Backbone(
            embed_dim, embed_dim, num_views, 1.0, 
            dropout, num_heads, num_cancer_types
        )
        
        # --- Step 4: Final Fusion and Classification ---
        # (h_unified + h_gnn)
        combined_dim_final = embed_dim + (embed_dim * num_views) 
        self.fusion_gate = nn.Sequential(
            nn.Linear(combined_dim_final, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, combined_dim_final), nn.Sigmoid()
        )
        self.fusion_projection = nn.Linear(combined_dim_final, embed_dim)
        
        self.cancer_embedding = nn.Embedding(num_cancer_types, embed_dim)
        self.context_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, 
            dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim // 2, 1)
        )
        self.manifold = geoopt.PoincareBall()

    def forward(self, omics_features, esm_features, gene_set_matrix, 
              edge_indices, edge_scores, cancer_type_id, causal_scores):
        
        # 1. Encode multi-modal features
        h_omics = self.omics_encoder(omics_features)
        h_esm = self.esm_encoder(esm_features)
        h_func = self.func_encoder(gene_set_matrix)
        
        # 2. Fuse modalities
        h_unified, self.attention_weights = self.fuser(h_omics, h_esm, h_func)

        # 3. Pass fused features through the GNN backbone
        # gnn_embedding_tan: For classification (Euclidean/tangent space)
        # gnn_embedding_hyp_list: For contrastive learning (hyperbolic space)
        gnn_embedding_tan, gnn_embedding_hyp_list = self.gnn_backbone(
            h_unified, edge_indices, edge_scores, cancer_type_id, causal_scores
        )
        
        # 4. Dual-path combination (skip-connection)
        combined_representation = torch.cat([h_unified, gnn_embedding_tan], dim=1)
        
        # 5. Gated fusion
        gate_values = self.fusion_gate(combined_representation)
        gated_representation = combined_representation * gate_values
        base_representation = self.fusion_projection(gated_representation)
        
        # 6. Apply context
        final_representation = base_representation
        if cancer_type_id is not None:
            cancer_context = self.cancer_embedding(cancer_type_id).expand(base_representation.shape[0], -1)
            transformer_input = torch.stack([base_representation, cancer_context], dim=1)
            transformer_output = self.context_transformer(transformer_input)
            final_representation = transformer_output[:, 0]

        # 7. Final classification
        logits = self.classifier(final_representation)
        
        if self.training:
            # --- Calculate the average of hyperbolic embeddings (for contrastive loss) ---
            # Stack hyperbolic embeddings from all views
            stacked_hyp_embeddings = torch.stack(gnn_embedding_hyp_list, dim=0)
            # Map all points to the tangent space at origin (logmap0)
            stacked_tan_embeddings = self.manifold.logmap0(stacked_hyp_embeddings)
            # Calculate the standard Euclidean mean in the tangent space
            avg_tan_embedding = torch.mean(stacked_tan_embeddings, dim=0)
            # Map the mean vector back to hyperbolic space (expmap0)
            avg_hyp_embedding = self.manifold.expmap0(avg_tan_embedding)
            
            return logits.squeeze(-1), avg_hyp_embedding
        else:
            return logits.squeeze(-1), None